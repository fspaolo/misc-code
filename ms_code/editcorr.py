#!/usr/bin/env python

#    editcorr.py - Script para rodar o programa de edicao e 
#    correcao (em C++) dos dados de altimetria no formato GDR ASCII.
#    
#    Uso: python <este_prog.py> [sat] <arq_entrada> <arq_geoid> <arq_saida>
#    onde
#        sat = seasat ou geogm ou ers1gm
#    
#    Author:	Fernando Paolo 
#    Date:		18-Set-2007.


from os import system, path
from sys import argv, exit
from shutil import move
from string import replace

# Programa de edicao (arq C++ compilado).
prog = 'editcorr'

# Programs de saida de editcorr.cc.
editout  = 'editcorr.out'
editinfo = 'editcorr.info'


# Verifica os argumentos
if len(argv) != 5:
    print 'Uso: python %s [sat] <arq_entrada> <arq_geoid> <arq_saida>' \
           % argv[0]
    print 'sat = seasat ou geogm ou ers1gm'
    exit()

# Verifica se existem os arqs de entrada.
if not path.exists(argv[2]):
    print 'Falta o arq de entrada: %s' % argv[2]
    exit()
if not path.exists(argv[3]):
    print 'Falta o arq de entrada: %s' % argv[3]
    exit()

# Verifica se existe o prog de leitura.
if not path.exists(prog):
    print 'Falta o programa de leitura: %s' % prog
    print 'Compilando %s.cc...' % prog
    try:
        system('g++ -ansi -Wall %s.cc -o %s' % (prog, prog)) 
	if not path.exists(prog):
	    print 'O prog %s.cc nao foi compilado!' % prog
	    exit()
    except:
        print 'Erro durante a compilacao!'
        exit()

# Passa os argumentos.
sat    = argv[1]
fdata  = argv[2]
fgeoid = argv[3]
fout   = argv[4]

if sat == 'seasat':
    print 'Editando Seasat...'
elif sat == 'geogm':
    print 'Editando Geosat GM...'
elif sat == 'ers1gm':
    print 'Editando ERS-1 GM...'

# Roda o prog de edicao e correcao.
system("./%s %s %s %s" % (prog, sat, fdata, fgeoid))

# Renomeia e move os arqs de saida para o dir dado.
try:
    move(editout, fout)
    move(editinfo, fout.replace('.txt', '.info'))
except:
    print 'Algum erro com os arqs de saida!'
    exit()

print 'Arquivos de saida:'
print '%s' % fout
print '%s' % fout.replace('.txt', '.info')
