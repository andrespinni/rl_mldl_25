#!/bin/bash

# Controlla se è stato passato un messaggio di commit come argomento
if [ $# -eq 0 ]; then
    echo "Errore: devi fornire un messaggio di commit."
    echo "Esempio: ./commit_wandb_fix.sh 'Messaggio di commit'"
    exit 1
fi

# Prendi il messaggio di commit dal primo argomento
commitMessage="$1"

# Elimina la cartella wandb/ se esiste
if [ -d "./wandb" ]; then
    rm -rf "./wandb"
    echo "Cartella 'wandb/' eliminata con successo."
else
    echo "Cartella 'wandb/' non presente, nessuna eliminazione necessaria."
fi

# Aggiunge tutti i file modificati e tracciati
git add -u

# Aggiunge nuovi file (escludendo wandb se ignorata)
git add .

# Effettua il commit
git commit -m "$commitMessage"

# Recupera il nome del ramo corrente
branch=$(git rev-parse --abbrev-ref HEAD)

# Esegue il push sul ramo corrente
git push origin $branch

echo "Commit e push completati sul ramo '$branch'."