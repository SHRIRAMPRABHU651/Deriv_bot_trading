# Running DERIVBOT as a Windows service with NSSM

1. **Install Python 3.11+** from python.org (tick *Add python.exe to PATH*).
2. **Create the virtual environment** (PowerShell):
   ```powershell
   cd C:\derivbot
   python -m venv .venv
   .\.venv\Scripts\python -m pip install -r requirements.txt
   ```
3. **Secrets:** copy `.env.example` to `C:\derivbot\.env` and fill in the DEMO values. The app reads
   `.env` itself; restrict its ACL to your account (`icacls .env /inheritance:r /grant:r "%USERNAME%:F"`).
   Copy `config.example.yaml` to `config.yaml` if you want non-default settings.
4. **Install NSSM** (https://nssm.cc) and register the service (elevated prompt):
   ```powershell
   nssm install DerivBot "C:\derivbot\.venv\Scripts\python.exe" "-m app.main"
   nssm set DerivBot AppDirectory C:\derivbot
   nssm set DerivBot AppEnvironmentExtra AUTOSTART=false
   nssm set DerivBot AppStdout C:\derivbot\logs\service.out.log
   nssm set DerivBot AppStderr C:\derivbot\logs\service.err.log
   nssm set DerivBot AppRotateFiles 1
   nssm set DerivBot AppRotateBytes 5000000
   nssm set DerivBot AppExit Default Restart
   nssm set DerivBot AppRestartDelay 5000
   nssm set DerivBot AppStopMethodConsole 20000    # lets the graceful shutdown finish
   nssm set DerivBot Start SERVICE_AUTO_START
   nssm start DerivBot
   ```
5. **Logs:** application logs are rotating JSON in `C:\derivbot\logs\derivbot.log`; service
   stdout/stderr are in the files configured above.
6. **Restart / stop:** `nssm restart DerivBot`, `nssm stop DerivBot`, `nssm status DerivBot`.
7. **Prevent sleep** (a sleeping PC drops the WebSocket and misses settlements):
   ```powershell
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   ```
   Also disable "Allow the computer to turn off this device to save power" for your network adapter.
8. The dashboard is at http://127.0.0.1:8000. It binds to localhost only; do not expose it.

The bot always starts in **DEMO** and stopped (`AUTOSTART=false`). Open the dashboard, enter the
dashboard token (printed once in the log/console if you did not set `DASHBOARD_TOKEN`), and press Start.
