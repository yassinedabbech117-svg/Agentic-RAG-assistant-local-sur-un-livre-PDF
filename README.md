# Agentic RAG — assistant local sur un livre PDF

Application de questions-réponses fondée sur le livre **The Hitchhiker’s Guide to Agentic AI**. Elle extrait le texte du PDF, retrouve les passages utiles avec une recherche hybride, puis demande à un modèle Ollama local de rédiger une réponse courte avec des références vérifiées aux pages récupérées.

Le PDF du livre, fourni sous licence de redistribution, est inclus à la racine du dépôt sous le nom `The Hitchhiker’s Guide to Agentic AI.pdf`.

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Utiliser le chatbot](#utiliser-le-chatbot)
- [Fonctionnement](#fonctionnement)
- [Prérequis](#prérequis)
- [Installation et démarrage](#installation-et-démarrage)
- [Première utilisation](#première-utilisation)
- [Configuration](#configuration)
- [Évaluation](#évaluation)
- [Dépannage](#dépannage)
- [Fichiers du projet](#fichiers-du-projet)

## Fonctionnalités

- Interface de discussion web fournie par Streamlit.
- Recherche hybride : mots-clés BM25 et recherche sémantique par embeddings.
- Similarité cosinus calculée par produit scalaire de vecteurs normalisés dans FAISS.
- Fusion des classements lexical et sémantique avec RRF (Reciprocal Rank Fusion).
- Génération locale par Ollama, avec `qwen3.5:0.8b` par défaut.
- Références aux pages limitées aux pages des passages réellement récupérés.
- Découpage actuel : **600 tokens**, avec **120 tokens** de chevauchement.
- Aucun service d’IA distant n’est requis pour les questions : Ollama fonctionne sur l’ordinateur local.

## Utiliser le chatbot

1. Démarrez Ollama, puis l’application web en suivant la section [Installation et démarrage](#installation-et-démarrage).
2. Ouvrez l’adresse locale affichée dans le terminal, en général `http://localhost:8501`.
3. Saisissez une question sur le livre dans la zone de discussion, puis envoyez-la.
4. Attendez la recherche et la génération. La première question peut être plus lente parce que le PDF et l’index sémantique sont préparés.
5. Lisez la réponse. Ouvrez **Sources** sous celle-ci pour consulter les passages et les pages qui ont été transmis au modèle.
6. Dans la barre latérale, choisissez le nombre de passages à fournir au modèle et le modèle Ollama installé. Pour le réglage évalué ici, gardez **5 passages** et `qwen3.5:0.8b`.
7. Utilisez **Clear conversation** pour effacer l’historique de la conversation.

Posez des questions précises et autonomes, par exemple : « What are the three modes used by Adaptive RAG? » Si le livre ne contient pas la réponse dans les passages trouvés, l’assistant doit le signaler plutôt que compléter avec des connaissances extérieures. Vérifiez les références dans le panneau des sources lorsque la précision est importante.

## Fonctionnement

1. **Extraction** — PyMuPDF extrait le texte page par page du PDF situé à la racine du projet.
2. **Découpage** — le tokenizer Qwen découpe le texte en fenêtres de 600 tokens, avec 120 tokens partagés entre fenêtres voisines. Les pages de chaque fenêtre sont conservées.
3. **Recherche BM25** — classe les fenêtres selon leurs termes communs avec la question. Cette recherche aide pour les noms propres, les acronymes et les termes exacts.
4. **Recherche sémantique** — Ollama calcule l’embedding de la question. FAISS compare ce vecteur normalisé aux vecteurs normalisés des passages ; leur produit scalaire correspond à la similarité cosinus.
5. **Fusion RRF** — combine les rangs BM25 et sémantiques sans additionner directement leurs scores, dont les échelles diffèrent.
6. **Génération** — les passages les mieux classés sont envoyés à Ollama. Le prompt limite la réponse aux faits présents dans ces extraits et demande une réponse directe et courte. Les références `[S1]`, `[S2]`, etc. sont converties en pages des passages correspondants; les références de page inventées par le modèle sont retirées.

L’index sémantique est calculé une première fois puis conservé dans `rag_data/`. Les passages mis en cache sont associés au PDF et aux paramètres de découpage; un changement détecté provoque leur reconstruction ainsi que celle de l’index. FAISS utilise la version CPU (`faiss-cpu`), tandis qu’Ollama choisit le processeur ou le GPU disponibles pour ses modèles.

## Prérequis

- Windows 10 ou version ultérieure.
- Python 3.11 recommandé.
- Ollama installé et démarré.
- Le PDF `The Hitchhiker’s Guide to Agentic AI.pdf` à la racine du dossier.
- Le fichier tokenizer Qwen requis au premier découpage.
- De l’espace disque pour l’environnement Python et les modèles Ollama.

## Installation et démarrage

Ouvrez PowerShell dans le dossier du projet.

### 1. Installer Ollama et les modèles

Installez Ollama depuis [ollama.com/download](https://ollama.com/download), puis ouvrez l’application Ollama. Dans PowerShell, récupérez le modèle de génération et celui des embeddings :

```powershell
ollama pull qwen3.5:0.8b
ollama pull qwen3-embedding:0.6b
ollama list
```

Le modèle d’embeddings est utilisé pour la recherche sémantique; `qwen3.5:0.8b` rédige les réponses. Les modèles `qwen3.5:2b` et `qwen3.5:4b` proposés dans l’interface sont facultatifs : téléchargez-les avant de les sélectionner.

L’application Ollama de Windows lance normalement son serveur automatiquement. Si l’API locale ne répond pas, ouvrez Ollama. Ne lancez `ollama serve` que si aucun serveur Ollama n’est déjà actif.

### 2. Installer le tokenizer Qwen

Le tokenizer sert à mesurer les tailles de chunks en tokens; il ne télécharge pas les poids du modèle de génération.

```powershell
New-Item -ItemType Directory -Force rag_data\qwen_tokenizer | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json" -OutFile "rag_data\qwen_tokenizer\tokenizer.json"
```

### 3. Installer les dépendances Python

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
```

### 4. Démarrer l’application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Ouvrez l’adresse indiquée par Streamlit, généralement [http://localhost:8501](http://localhost:8501). Gardez le terminal ouvert pendant l’utilisation. Pour arrêter l’application, revenez au terminal et faites `Ctrl+C`.

## Première utilisation

À la première question, l’application extrait le PDF, crée les chunks et demande à Ollama les embeddings de tous les passages. Cette indexation peut prendre plusieurs minutes, notamment si Ollama fonctionne sur le processeur. Elle est mise en cache pour les lancements suivants. Les questions suivantes ne recalculent que l’embedding de la question, lancent BM25 et FAISS, puis génèrent la réponse.

Si le PDF change ou si vous changez `CHUNK_SIZE`/`CHUNK_OVERLAP`, l’application détecte que le cache ne correspond plus et reconstruit les chunks et l’index. Les données générées restent dans `rag_data/` et les fichiers d’index locaux ne sont pas nécessaires dans Git.

### Mode console

Pour utiliser le RAG directement dans PowerShell :

```powershell
.\.venv\Scripts\python.exe main.py
```

Posez vos questions à l’invite `Question :`. Entrez `exit`, `quit` ou `q` pour quitter.

## Configuration

Les valeurs principales sont dans `config.py` :

| Paramètre | Valeur par défaut | Rôle |
| --- | --- | --- |
| `PDF_PATH` | PDF à la racine | Document de connaissances. |
| `CHUNK_SIZE` | `600` tokens | Taille des passages. |
| `CHUNK_OVERLAP` | `120` tokens | Tokens partagés entre fenêtres voisines. |
| `CHUNK_TOKENIZER_PATH` | `rag_data/qwen_tokenizer/tokenizer.json` | Tokenizer pour compter et découper les tokens. |
| `OLLAMA_MODEL` | `qwen3.5:0.8b` | Générateur par défaut; l’interface permet d’en choisir un autre installé. |
| `OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Modèle d’embeddings pour la recherche sémantique. |
| `BM25_TOP_K` | `8` | Candidats lexicaux avant fusion. |
| `SEMANTIC_TOP_K` | `8` | Candidats sémantiques avant fusion. |
| `FINAL_TOP_K` | `5` | Nombre par défaut de passages après fusion. |

Les adresses Ollama par défaut sont `http://127.0.0.1:11434/api/chat` et `http://127.0.0.1:11434/api/embed`.

## Évaluation

Le script `evaluate_rag_20q.py` exécute les mêmes 20 questions, avec la recherche hybride, cinq passages au modèle et le générateur défini dans `config.py`. Les tailles de chunks se choisissent avec les options :

```powershell
.\.venv\Scripts\python.exe evaluate_rag_20q.py --chunk-size 300 --overlap 50
.\.venv\Scripts\python.exe evaluate_rag_20q.py --chunk-size 600 --overlap 120
```

Chaque exécution produit `rag_data/rag_20_questions_<taille>_<overlap>.json`, contenant les passages récupérés, réponses, durées et notes de fidélité. Les fichiers de rapport déjà présents reprennent les résultats d’évaluations précédentes. L’évaluation de la récupération compare les pages des passages aux pages attendues; ce Hit@5 est un indicateur, pas une preuve complète de pertinence. Les notes de fidélité sont une appréciation manuelle et dépendent du jeu de 20 questions utilisé.

Un autre script, `evaluate_retrievers.py`, compare BM25, recherche sémantique et recherche hybride sur le jeu de questions de comparaison, pour les tailles 300/50 et 600/120. Il enregistre les résultats dans `rag_data/retriever_chunk_comparison.json`.

## Dépannage

| Problème | Solution |
| --- | --- |
| Ollama n’est pas accessible | Ouvrez Ollama et vérifiez `ollama list`. Le serveur doit répondre sur `127.0.0.1:11434`. |
| Modèle manquant | Installez le modèle demandé avec `ollama pull nom-du-modèle`. |
| La première question est très lente | Laissez l’indexation sémantique finir. Les questions suivantes réutilisent l’index. |
| Le tokenizer est introuvable | Reprenez l’étape 2 et vérifiez `rag_data\qwen_tokenizer\tokenizer.json`. |
| Le PDF est introuvable | Vérifiez le nom et l’emplacement, ou ajustez `PDF_PATH` dans `config.py`. |
| Le PDF ne fournit aucun texte | Il peut être composé d’images; appliquez-lui un OCR avant de l’utiliser. |
| La réponse est lente | Gardez un petit modèle, par exemple `qwen3.5:0.8b`, vérifiez l’activité CPU/GPU et le modèle utilisé avec `ollama ps`. |
| Streamlit utilise un autre port | Ouvrez l’URL exacte affichée dans le terminal. |
| Les passages ou résultats semblent périmés | Arrêtez puis relancez Streamlit. Le cache de chunks est régénéré si le PDF ou les paramètres ont changé. |

## Fichiers du projet

| Fichier | Rôle |
| --- | --- |
| `app.py` | Interface Streamlit, réglages de recherche, historique et affichage des sources. |
| `rag_service.py` | Relie la recherche hybride au générateur. |
| `main.py` | Charge ou crée les chunks et les index; contient aussi le mode console. |
| `pdf_loader.py` | Extrait le texte et les numéros de page du PDF. |
| `chunker.py` | Découpe le texte en fenêtres tokenisées et conserve les pages. |
| `hybrid_search.py` | BM25, recherche candidate et fusion des classements RRF. |
| `semantic_search.py` | Appels Ollama pour les embeddings et recherche FAISS par cosinus. |
| `generator.py` | Génération Ollama, réponses concises et validation des références de page. |
| `config.py` | Chemins, modèles, taille des chunks et paramètres de recherche. |
| `evaluate_rag_20q.py` | Évaluation complète sur 20 questions avec taille configurable. |
| `evaluate_retrievers.py` | Comparaison des méthodes de récupération et tailles de chunks. |
| `requirement.txt` | Dépendances Python. |
| `rag_data/` | Chunks, rapports, tokenizer local et index sémantique généré. |
