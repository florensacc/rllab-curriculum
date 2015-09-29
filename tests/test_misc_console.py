import sys
from misc.console import tweak, type_hint

def hello(name='world'):
    return 'hello, %s' % name

sys.argv = ['(program name)', '--hello.name', 'john']
print tweak(hello)()

@type_hint('name', str)
def hello2(name):
    return 'hello, %s' % name

print hello2.__tweak_type_hint_meta__

sys.argv = ['(program name)', '--hello2.name', 'john']
print tweak(hello2)('world')

sys.argv = ['(program name)', '--some_number', '3']
print tweak(2, 'some_number')
