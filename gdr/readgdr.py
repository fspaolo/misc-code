#!/usr/bin/env python
#------------------------------------------------------------------
# readgdr.py
# Script para rodar o programa de leitura (em C++) dos dados de 
# altimetria no formato GDR, em todos os arqs de uma so vez.
# 
# Author:	Fernando S. Paolo 
# Date:		18-Set-2007.
#------------------------------------------------------------------
#  
# Usage: 
#   python <este_prog.py> [gdr] [reg] [end] <dados> <arq_saida>
#
#   gdr = seasat|geogm|ers1gm
#   reg = lon1 lon2 lat1 lat2
#   end = 1 (Work-station) ou 0 (PC)

from glob import glob
from os import system, path
from sys import argv, exit
from shutil import move
from string import replace

# Programa de leitura (arq C++ compilado)
prog = 'readgdr'

# Programs de saida de readgdr.cc.
readout  = 'readgdr.out'
readinfo = 'readgdr.info'


# Verifica os argumentos
if len(argv) < 9:
    print 'Uso: python %s [gdr] [reg] [end] <dados> <arq_saida>' % argv[0]
    print 'gdr = seasat ou geogm ou ers1gm'
    print 'reg = lon1 lon2 lat1 lat2'
    print 'end = 1 (Work-station) ou 0 (PC)'
    exit()

# Verifica se existe o prog de leitura.
if not path.exists(prog):
    print 'Falta o programa de leitura: %s' % prog
    print 'Compilando %s.cc...' % prog
    try:
        system('g++ -ansi -Wall %s.cc -o %s' % (prog, prog)) 
    except:
        print 'Erro durante a compilacao!'
        exit()

# Passa os parametros.
gdr  = argv[1]
reg  = argv[2] + " " + argv[3] + " " + argv[4] + " " + argv[5]
end  = argv[6]
data = argv[7]
fout = argv[8]


if gdr == 'seasat':
    print 'Lendo Seasat...'
elif gdr == 'geogm':
    print 'Lendo Geosat GM...'
elif gdr == 'ers1gm':
    print 'Lendo ERS-1 GM...'
    
# Lista todos os arqs indicados em data
files = glob(data) 
files.sort()            # ordena os arqs

# converte a lista de arqs para string (com espacos)
strData = ""
for s in files:         
    strData += s + " "  

# roda o prog de leitura sobre todos os arqs binarios
system("./%s %s %s %s %s" % (prog, gdr, reg, end, strData))

# Renomeia e move os arqs de saida para o dir dado.
try:
    move(readout, fout)
    move(readinfo, fout.replace('.txt', '.info'))
except:
    print 'Algum erro com os arqs de saida!'
    exit()

print 'Arquivos de saida:'
print '%s' % fout
print '%s' % fout.replace('.txt', '.info')
