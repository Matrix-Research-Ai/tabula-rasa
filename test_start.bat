@echo on
cd /d "C:\Users\Admin\tabula-rasa"
echo Current dir: %cd%
echo Starting tabula rasa...
start "Tabula Rasa AI" /B pythonw -c "from egefalos.tabula_rasa import main; main()"
echo tabula rasa started, waiting...
timeout /t 5 /nobreak >nul
echo Checking netstat...
netstat -ano | find ":8002 "
echo Done.
