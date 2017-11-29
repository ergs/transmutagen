$PROJECT = 'transmutagen'
$ACTIVITIES = [
              'tag',  # Creates a tag for the new version number
              'pypi',  # Sends the package to pypi
              # 'conda_forge',  # Creates a PR into your package's feedstock
              'ghrelease'  # Creates a Github release entry for the new tag
               ]
$TAG_REMOTE = 'git@github.com:ergs/transmutagen.git'  # Repo to push tags to

$GITHUB_ORG = 'ergs'  # Github org for Github releases and conda-forge
$GITHUB_REPO = 'transmutagen'  # Github repo for Github releases  and conda-forge
