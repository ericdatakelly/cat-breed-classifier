1. Copy these files into a new project folder.  Exclude `.git`.  (or use this as a repo template in GitHub)
2. Update all files for the new project (project name, version, dependencies, etc.)
3. Create virtual environment (`mamba env create -f environment.yaml`) and activate
4. Install via `flit install -s`
5. In VS Code
  - Initialize repo
  - Select Python interpreter (might also need to be in settings.json)
  - If tests are present, discover tests using Test Explorer (tab) and run tests
6. Try debugging a file using VS Code debugger.  Click the button to make a launch file and accept defaults.  This will reduce clicking and will provide a place for options when needed.  Note that python interpreter should be shown by the full path in the terminal (it's not necessarily activated).
