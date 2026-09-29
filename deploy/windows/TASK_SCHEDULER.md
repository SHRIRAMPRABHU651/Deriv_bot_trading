# Running DERIVBOT with Windows Task Scheduler (no extra tools)

1. Install Python 3.11+, create the virtualenv and `.env` exactly as in [NSSM.md](NSSM.md) steps 1-3.
2. Create a launcher `C:\derivbot\run.cmd`:
   ```bat
   @echo off
   cd /d C:\derivbot
   set AUTOSTART=false
   .venv\Scripts\python.exe -m app.main >> logs\task.out.log 2>&1
   ```
3. Open **Task Scheduler → Create Task…**
   - *General*: name `DerivBot`, "Run whether user is logged on or not", "Run with highest privileges".
   - *Triggers*: **At startup** (and optionally *At log on*).
   - *Actions*: Start a program → `C:\derivbot\run.cmd`.
   - *Conditions*: untick "Start the task only if the computer is on AC power" and
     "Stop if the computer switches to battery power".
   - *Settings*: tick "If the task fails, restart every **1 minute**, up to **999** times";
     untick "Stop the task if it runs longer than…".
4. **Logs:** `C:\derivbot\logs\derivbot.log` (rotating JSON) and `C:\derivbot\logs\task.out.log`.
5. **Restart:** right-click the task → *End* then *Run*.
6. **Prevent sleep:** `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0`;
   keep the machine on AC power and disable network-adapter power saving.
7. Ending the task sends a console close; for the fully graceful shutdown path (reconcile → persist → close)
   prefer stopping from the dashboard (**Stop**) before ending the task, or use NSSM.
