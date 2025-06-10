#!/bin/bash
#Linux bash script using ImageMagick and perl-rename

for file in *; do mv "$file" `echo $file | tr ' ' '_'`; done                                                                                  
mogrify -format png -fuzz 0% -trim +repage *.png
for filename in *.png; do
    magick "$filename" -gravity Center -background White -extent "%[fx:max(w,h)]x%[fx:max(w,h)]" "$filename"
done
mogrify -format png -resize 500x500 *.png
mogrify -format png -bordercolor white -border 25 -threshold 50% *.png
#cp *.png ../500px   #store 500px versions of the images if needed
mogrify -format png -resize 32x32 *.png
perl-rename 'y/A-Z/a-z/' *
#remove color profile that the new Gimp version appears to insert
for image in *.png; do magick "$image" -strip -type Grayscale "$image"; done
