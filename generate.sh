cd latex

pandoc -s main.tex -o ../index.html --self-contained --number-sections --citeproc

cd ../

sed -i 's/startaudio /<audio controls src="/' index.html
sed -i 's/endaudio/"><\/audio>/' index.html