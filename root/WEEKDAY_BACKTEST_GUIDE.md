# Ugedagskalender

Den fælles kalender kan planlægge fem tidligere uger og en prognoseuge:

```powershell
python -m src.exe.run_model.weekday_backtest --first-origin 2022-03-28 --last-origin 2022-04-04 --output kalender --plan-only
```

Modelkørsel kræver en egen --adapter module:zero_argument_factory samt
--history-csv og --inputs-csv. Kontrakten står i
[kalenderkoden](../src/tools/weekday_backtest.py).
G's grundmodel køres efter [BM-guiden](BM_GUIDE.md).
G's tidligere studieadaptere er bevaret på G-arkivbranchen.
