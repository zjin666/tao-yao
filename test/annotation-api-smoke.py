"""One synthetic utterance, real API; never inserts probe into collected samples."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from taoyao.annotation import DeepSeekJudge, atomic_json, contract
from taoyao.config import load_config

async def main():
    config=load_config()
    judge=DeepSeekJudge({**config['deepseek'],**config['language_analysis']})
    try:
        result,reasons=await judge.analyze('你是不是有点累了？',[],contract()[1])
        path=Path(config['language_analysis']['directory'])/'probes/api_smoke.json'
        atomic_json(path,dict(source='synthetic_api_probe_NOT_training',text='你是不是有点累了？',primary=result,reasons=reasons))
        print('API OK:',result['model_returned'])
        print({d:r['bucket'] for d,r in result['labels'].items()})
    except Exception as exc:
        atomic_json(Path(config['language_analysis']['directory'])/'probes/api_failure.json',
            dict(source='synthetic_probe_NOT_training',error_type=type(exc).__name__,payload=getattr(judge,'last_payload',None)))
        raise
    finally:
        await judge.close()

if __name__=='__main__':asyncio.run(main())
