"""python -m taoyao.annotation_cli --help"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path

from .annotation import SampleStore, DeepSeekJudge, atomic_json, contract
from .config import load_config
from .personality import DIMENSIONS


def queue(store, limit):
    rows=store.all()
    counts=Counter((d,v['bucket']) for r in rows for d,v in (r.get('primary') or {}).get('labels',{}).items())
    pending=[r for r in rows if r['human_review']['status']=='pending']
    def rank(row):
        reasons=row['review_reasons']
        rare=any(v['bucket'] not in ('unknown','中性') and counts[d,v['bucket']]<20
            for d,v in (row.get('primary') or {}).get('labels',{}).items())
        return 100*('primary_failed' in reasons)+50*any('uncertain' in s for s in reasons)+30*any('disagreement' in s for s in reasons)+20*rare+10*('random_audit' in reasons)
    pending.sort(key=lambda r:(-rank(r),r['collected_at'],r['id']))
    items=[]
    for row in pending[:limit]:
        labels={d:(row.get('primary') or {}).get('labels',{}).get(d,{}).get('bucket','unknown') for d in DIMENSIONS}
        items.append(dict(sample_id=row['id'],text=row['text'],raw_text=row['raw_text'],context=row['context'],
            sent_at=row['sent_at'],timezone=row['timezone'],scene=row['scene'],definition_version=row['definition_version'],
            reasons=row['review_reasons'],primary=row['primary'],local=row['local'],
            decision='pending',labels=labels,reviewer='',note=''))
    # Timestamped editable queues: generating another queue never erases unfinished edits.
    from datetime import datetime,timezone
    path=store.root/'review'/('queue-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    atomic_json(path,dict(instructions='Edit decision to confirm or reject; check all six labels, use unknown if unsure; fill reviewer. pending is ignored.',items=items))
    print(path)


def apply(store, path):
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    count=0
    for item in payload['items']:
        decision=item['decision']
        if decision=='pending':continue
        if decision not in ('confirm','reject'):raise ValueError('decision must be pending/confirm/reject')
        row=store.read(item['sample_id'])
        if row['definition_version']!=item['definition_version']:raise ValueError('Definition mismatch')
        store.review(row['id'],item['labels'],item['reviewer'],item.get('note',''),reject=decision=='reject')
        count+=1
    print(f'Applied {count} human reviews. No training or live score changes.')


def export(store, version, output):
    rows=[r for r in store.all() if r['human_review']['status']=='confirmed' and r['definition_version']==version]
    # Union connected conversations and exact normalized texts before splitting.
    # All messages in a private user / whole group stay together (conservative).
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)]=min(a,b)
    def conversation(r):
        return r['scene']+':'+(r['group_id'] if r['scene']=='group' else r['user_id'])
    for r in rows:
        normalized=''.join(c.lower() for c in r['text'] if c.isalnum())
        if not normalized:normalized=r['text']
        union('c:'+conversation(r),'t:'+hashlib.sha256(normalized.encode()).hexdigest())
    result=[];seen=set()
    for r in rows:
        group=hashlib.sha256(find('c:'+conversation(r)).encode()).hexdigest()
        number=int(group[:8],16)%100
        split='train' if number<80 else 'dev' if number<90 else 'test'
        for dim,bucket in r['human_review']['labels'].items():
            if bucket=='unknown':continue
            # Identical messages do not gain extra weight just by repeated sending.
            key=(r['text'],dim,bucket)
            if key in seen:continue
            seen.add(key)
            result.append(dict(sample_id=r['id'],text=r['text'],context=r['context'],dimension=dim,bucket=bucket,
                split=split,group_id=group,definition_version=version,label_origin='human_confirmed',
                sent_at=r['sent_at'],timezone=r['timezone']))
    # Contradictory labels for identical text may reflect context; leave visible, never auto-resolve.
    counts=Counter(r['split'] for r in result)
    atomic_json(output,dict(definition_version=version,rows=result,split_counts=dict(counts),
        warning='Only human-confirmed known axes. Conversation/text connected components are isolated. Small corpora may lack dev/test; do not force split conversations. Context-dependent labels retained. No automatic training.'))
    print(f'Exported {len(result)} axis labels to {output}; split_counts={dict(counts)}')


async def retry(store, config, sid, model):
    row=store.read(sid)
    cfg={**config['deepseek'],**config['language_analysis']}
    if model:cfg['model']=model
    spec=json.loads((store.root/'definitions'/f"{row['definition_version']}.json").read_text(encoding='utf-8'))
    judge=DeepSeekJudge(cfg)
    try:
        primary,reasons=await judge.analyze(row['text'],row['context'],spec)
        local=row.get('local') or {}
        for dim,item in primary['labels'].items():
            if (local.get(dim,{}).get('candidate') or local.get(dim,{}).get('bucket'))!=item['bucket']:
                reasons.append(f'{dim}:local_disagreement')
            if item['bucket'] not in ('unknown','中性'):reasons.append(f'{dim}:signal_review')
        if int(sid[:8],16)%10==0:reasons.append('random_audit')
        await asyncio.to_thread(store.analysis,sid,primary,local,reasons)
    finally:
        await judge.close()
    print('Analysis saved; previous analysis and human reviews preserved. No live score changes.')


def main():
    parser=argparse.ArgumentParser(description='Local sample review pipeline; retry alone calls DeepSeek')
    parser.add_argument('--config')
    parser.add_argument('--directory')
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('stats')
    p=sub.add_parser('queue');p.add_argument('--limit',type=int,default=20)
    p=sub.add_parser('apply');p.add_argument('file')
    p=sub.add_parser('export');p.add_argument('--version');p.add_argument('--output')
    p=sub.add_parser('retry');p.add_argument('sample_id');p.add_argument('--model')
    args=parser.parse_args()
    config=load_config(args.config)
    store=SampleStore(args.directory or config['language_analysis']['directory'])
    if args.action=='stats':
        rows=store.all()
        print(json.dumps(dict(samples=len(rows),review_status=dict(Counter(r['human_review']['status'] for r in rows)),
            analysis_status=dict(Counter((r.get('primary') or {}).get('status','pending') for r in rows)),
            dimensions={d:dict(Counter(r['human_review']['labels'].get(d,'unknown') for r in rows if r['human_review']['status']=='confirmed')) for d in DIMENSIONS}),ensure_ascii=False,indent=2))
    elif args.action=='queue':queue(store,max(1,args.limit))
    elif args.action=='apply':apply(store,args.file)
    elif args.action=='export':
        version=args.version or contract()[0]
        export(store,version,Path(args.output) if args.output else store.root/'exports'/f'human-{version}.json')
    elif args.action=='retry':asyncio.run(retry(store,config,args.sample_id,args.model))


if __name__=='__main__':main()
