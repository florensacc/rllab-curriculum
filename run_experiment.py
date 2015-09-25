#!/usr/bin/python

import algo
import sys
from misc.click import CLASS
import traceback

#@click.command()
#@click.option('--algorithm', default='algo.TRPO', help='Algorithm to use.')
#@click.option('--name', help='The person to greet.')
#def run_experiment(algorithm,name):
#    algo = algorithm()
#    mdp = mdp()
    #return None
    #for x in range(count):
    #    click.echo('Hello %s!' % name)

#if __name__ == '__main__':
#    import inspect
#    print inspect.getargspec(algo.TRPO.__init__)
    #import argparse

    #parser = argparse.ArgumentParser(description='Run experiment')
    #parser.add_argument('algorithm', action="store", type=int)
    #parser.add_argument('units', action="store")

    #print parser.parse_args()
    #oldstdin = sys.stdin.write# = None
    #oldstdout = sys.stdout.write# = None
    #oldstderr = sys.stderr.write
    #def a(*args):
    #    print 'herea'; return oldstdout(*args)
    #def b(*args):
    #    print 'hereb'; return oldstderr(*args)

    #sys.stdout.write = a
    #sys.stderr.write = b
    #try:
    #    run_experiment()
    #except Exception as e:
    #    traceback.print_exc()
    #    print 'ex'
