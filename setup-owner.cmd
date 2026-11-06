@echo off
cd /d "%~dp0"
echo Enter ONLY your QQ number. Do NOT enter your QQ password.
python -c "import json,pathlib; p=pathlib.Path('config.json'); d=json.loads(p.read_text('utf-8-sig')); q=input('Owner QQ: ').strip(); d['owner_qq']=int(q) if q.isdigit() else 0; p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8'); print('saved' if q.isdigit() else 'invalid, not saved')"
pause
