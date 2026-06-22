# PR INSTRUCTIONS

1. Create PR
  - IF branch == dev: to main
  - ELSE: to dev, ~~gh action SHOULD automatically change it to dev if set to main from non-dev branch (see .github/workflows/auto-change-base.yml)~~

2. Assign person

3. IF changes are possibly breaking: wait for approval

4. Squash & Merge

5. Delete branch merging FROM (INCLUDING dev if dev -> main)

6. IF dev -> main: create new dev branch from main (this keeps history clean and consistent)
