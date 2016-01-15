find . -type f -name "*.py" -print0 | LC_ALL=C LC_CTYPE=C LANG=C xargs -0 sed -i "" "s/import tensorfuse as theano, tensorfuse.tensor as TT/import theano, theano.tensor as TT/g"
find . -type f -name "*.py" -print0 | LC_ALL=C LC_CTYPE=C LANG=C xargs -0 sed -i "" "s/import tensorfuse as theano/import theano/g"
find . -type f -name "*.py" -print0 | LC_ALL=C LC_CTYPE=C LANG=C xargs -0 sed -i "" "s/import tensorfuse.tensor as T/import theano.tensor as T/g"
find . -type f -name "*.py" -print0 | LC_ALL=C LC_CTYPE=C LANG=C xargs -0 sed -i "" "s/import import tensorfuse.tensor as TT/theano.tensor as TT/g"
