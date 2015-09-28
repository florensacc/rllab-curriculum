import sys
from misc.console import tweak, type_hint

def hello(name='world'):
    return 'hello, %s' % name

sys.argv = ['*', '--hello.name', 'john']
print tweak(hello)()

@type_hint('name', str)
def hello2(name):
    return 'hello, %s' % name

print hello2.__tweak_type_hint_meta__

sys.argv = ['*', '--hello2.name', 'john']
print tweak(hello2)('world')
