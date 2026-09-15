# Marketing AI Project

## Analyse et optimisation marketing basée sur la segmentation client et l'intelligence artificielle

### Objectif

Exploiter les données clients, produits, ventes et marketing afin de :

- analyser les comportements d'achat ;
- segmenter les clients ;
- construire des personas ;
- évaluer les campagnes marketing ;
- prédire le churn ;
- estimer la valeur client ;
- proposer des recommandations marketing personnalisées ;
- développer un dashboard interactif.

## Données

- Customers
- Products
- Sales
- Marketing

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Git / GitHub
- Power BI ou dashboard HTML/Plotly

Pourquoi il manque le client 2005 (Eva) dans le résultat ?

Bonne question à se poser ! Regarde tes données sales : personne n'a jamais acheté avec Customer_ID = 2005. Comme on a construit features_clients à partir de df (la table fusionnée avec sales), et qu'Eva n'a aucune vente, elle disparaît automatiquement du groupby. C'est un point à noter dans ton rapport : "Eva (2005) n'a pas encore d'historique d'achat, elle est donc exclue de la segmentation basée sur le comportement d'achat."