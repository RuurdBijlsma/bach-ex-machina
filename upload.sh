./generate.sh

# Commit and upload to GitHub
git add -A
git commit -am 'Update GitHub pages website'
git pull origin gh-pages
git push origin gh-pages