find . -type f -name "*.py" -print0 | LC_ALL=C LC_CTYPE=C LANG=C xargs -0 sed -i "" "s/import theano.tensor as TT/import tensorfuse.tensor as TT/g"
find . -type f -name "*.py" -print0 | LC_ALL=C LC_CTYPE=C LANG=C xargs -0 sed -i "" "s/import theano.tensor as T/import tensorfuse.tensor as T/g"
find . -type f -name "*.py" -print0 | LC_ALL=C LC_CTYPE=C LANG=C xargs -0 sed -i "" "s/import theano/import tensorfuse as theano/g"
find . -type f -name "*.py" -print0 | LC_ALL=C LC_CTYPE=C LANG=C xargs -0 sed -i "" "s/import theano, theano.tensor as TT//g"
