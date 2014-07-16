#!/usr/bin/env python

import ftplib     # protocolo FTP
import os, sys    # modulos sistema operacional

#--------------------------------------------------------------

ftpServer = 'iliad.gsfc.nasa.gov' 
name      = 'anonymous' 
password  = 'fspaolo@iag.usp.br'

# Define o dir remoto onde estao os dados
remoteDir = '/opf/geosat/GM'

# Define o dir local onde serao baixados os dados
localDir  = "/home/fspaolo/DATA/sat_data/geosat/GM"

#--------------------------------------------------------------

# Se o dir nao existe cria um localmente
if not os.path.isdir(localDir):
    os.mkdir(localDir)

# Muda o dir local para onde serao baixados os dados
os.chdir(localDir)

# Acesando o servidor FTP
ftp = ftplib.FTP(ftpServer)  #, name, password)
print ftp.getwelcome()

# Login no servidor FTP
ftp.login(name, password)

# Indo para o diretorio remoto 
ftp.cwd(remoteDir)

# Lista o diretorio
directorys = ftp.nlst()

# Varre todos os diretorios
for dir in directorys:      

    remotedir = os.path.join(remoteDir, dir)
    localdir  = os.path.join(localDir, dir)
    eh_dir = 'True'

    # Entra em todos os dirs recusivamente
    while eh_dir == 'True':

        # Tenta entrar se for diretorio
        try:
            # Muda para o diretorio remoto
            ftp.cwd(remotedir)
	    print 'Entrando -> ', remotedir

            # Se nao existe cria o diretorio local
            if not os.path.isdir(localdir):
                os.mkdir(localdir)
            
            # Muda o dir local onde serao baixados os dados
            os.chdir(localdir)
            
            # Lista o diretorio
            folder = ftp.nlst()
            
            # todos os arquivos
            for file in folder:        

		# Tenta entrar se for diretorio
                try:
                    # Muda para o diretorio remoto
                    ftp.cwd( os.path.join(remotedir, file) )
                    remotedir = os.path.join(remotedir, file)
                    localdir  = os.path.join(localdir, file)

	            print 'Entrando -> ', remotedir

                    # Se nao existe cria o diretorio local
                    if not os.path.isdir(localdir):
                        os.mkdir(localdir)
                    
                    # Muda o dir local onde serao baixados os dados
                    os.chdir(localdir)
                    
                    # Lista o diretorio
                    files = ftp.nlst()
                    
		    # Baixa todos os arqs do dir
		    for fs in files:
		        print 'Baixando: %s' % os.path.join(remotedir, fs)
                        f = open(fs, 'wb')
			ftp.retrbinary('RETR %s' % fs, f.write)
                        f.close()
		        eh_dir = 'False'
            
		# Se for arq baixa, sai do loop recursivo
		except:
                    print 'Baixando: %s' % os.path.join(remotedir, file)
                    f = open(file, 'wb')
		    ftp.retrbinary('RETR %s' % file, f.write)
                    f.close()
		    eh_dir = 'False'

	# Se for arq baixa, sai do loop recursivo
        except:
            print 'Baixando: %s' % os.path.join(remotedir, dir)
            f = open(dir, 'wb') 
	    ftp.retrbinary('RETR %s' % dir, f.write)
            f.close()
	    eh_dir = 'False'

# Fecha a coneccao FTP
ftp.close()
