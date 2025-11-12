# AZDA FATIMA-ZAHRA
<img src="faat.jpg" style="height:200px;margin-right:150px"/>
# Etudiante en 4éme année ENCG filière contrôle-audit conseil
# Rapport d'Analyse : Dataset Default of Credit Card Clients

**Date du rapport :** 12 novembre 2025  
**Source :** UCI Machine Learning Repository  
**DOI :** 10.24432/C55S3H  

---

## 1. Résumé Exécutif

Ce rapport présente une analyse détaillée du dataset "Default of Credit Card Clients" disponible sur le dépôt UCI Machine Learning. Ce dataset vise à prédire le défaut de paiement des clients de cartes de crédit à Taiwan et compare l'exactitude prédictive de six méthodes de data mining différentes.

**Points clés :**
- 30 000 instances de clients
- 23 variables explicatives
- Variable cible binaire (défaut de paiement : Oui/Non)
- Données collectées à Taiwan en 2005
- Aucune valeur manquante

---

## 2. Contexte et Objectifs

### 2.1 Objectif de la Recherche

Cette recherche vise à analyser les cas de défauts de paiement des clients à Taiwan et à comparer la précision prédictive de la probabilité de défaut entre six méthodes de data mining. Du point de vue de la gestion des risques, la probabilité estimée de défaut est plus précieuse qu'une simple classification binaire (client fiable ou non fiable).

### 2.2 Innovation Méthodologique

L'étude présente la nouvelle méthode "Sorting Smoothing Method" pour estimer la probabilité réelle de défaut. Les résultats montrent que parmi les six techniques de data mining étudiées, le réseau de neurones artificiels est le seul capable d'estimer avec précision la probabilité réelle de défaut.

---

## 3. Caractéristiques du Dataset

### 3.1 Informations Générales

| Caractéristique | Détail |
|----------------|--------|
| **Type de dataset** | Multivarié |
| **Domaine** | Business / Finance |
| **Tâche associée** | Classification |
| **Type de features** | Integer, Real |
| **Nombre d'instances** | 30 000 |
| **Nombre de features** | 23 |
| **Valeurs manquantes** | Non |
| **Date de donation** | 25 janvier 2016 |
| **Taille du fichier** | 5.3 MB (.xls) |

### 3.2 Licence

Le dataset est sous licence **Creative Commons Attribution 4.0 International (CC BY 4.0)**, permettant le partage et l'adaptation pour tout usage avec attribution appropriée.

---

## 4. Description des Variables

### 4.1 Variable Cible

**Default Payment (Y)** : Variable binaire indiquant le défaut de paiement
- 1 = Oui (défaut)
- 0 = Non (pas de défaut)

### 4.2 Variables Démographiques (X1-X5)

| Variable | Description | Valeurs/Unité |
|----------|-------------|---------------|
| **X1 (LIMIT_BAL)** | Montant du crédit accordé | NT dollar |
| **X2 (SEX)** | Genre | 1 = masculin, 2 = féminin |
| **X3 (EDUCATION)** | Niveau d'éducation | 1 = études supérieures, 2 = université, 3 = lycée, 4 = autres |
| **X4 (MARRIAGE)** | État matrimonial | 1 = marié, 2 = célibataire, 3 = autres |
| **X5 (AGE)** | Âge | Années |

### 4.3 Historique des Paiements (X6-X11)

Variables traçant l'historique des paiements mensuels d'avril à septembre 2005 :

- **X6 (PAY_0)** : Statut de remboursement en septembre 2005
- **X7 (PAY_2)** : Statut de remboursement en août 2005
- **X8 (PAY_3)** : Statut de remboursement en juillet 2005
- **X9 (PAY_4)** : Statut de remboursement en juin 2005
- **X10 (PAY_5)** : Statut de remboursement en mai 2005
- **X11 (PAY_6)** : Statut de remboursement en avril 2005

**Échelle de mesure :**
- -1 = paiement à temps
- 1 = retard de 1 mois
- 2 = retard de 2 mois
- ...
- 8 = retard de 8 mois
- 9 = retard de 9 mois et plus

### 4.4 Montants des Factures (X12-X17)

Montant des relevés de facturation (NT dollar) d'avril à septembre 2005 :

- **X12** : Montant de la facture en septembre 2005
- **X13** : Montant de la facture en août 2005
- **X14** : Montant de la facture en juillet 2005
- **X15** : Montant de la facture en juin 2005
- **X16** : Montant de la facture en mai 2005
- **X17** : Montant de la facture en avril 2005

### 4.5 Montants des Paiements Précédents (X18-X23)

Montant des paiements effectués (NT dollar) d'avril à septembre 2005 :

- **X18** : Montant payé en septembre 2005
- **X19** : Montant payé en août 2005
- **X20** : Montant payé en juillet 2005
- **X21** : Montant payé en juin 2005
- **X22** : Montant payé en mai 2005
- **X23** : Montant payé en avril 2005

---

## 5. Résultats Principaux

### 5.1 Comparaison des Méthodes de Data Mining

L'étude a comparé six techniques de data mining pour prédire le défaut de paiement. Le réseau de neurones artificiels s'est révélé être la méthode la plus performante.

### 5.2 Performance du Réseau de Neurones

Utilisant une régression linéaire simple (Y = A + BX) où :
- Y = probabilité réelle de défaut
- X = probabilité prédite de défaut

Le réseau de neurones a montré :
- **Coefficient de détermination le plus élevé**
- **Intercept (A) proche de zéro**
- **Coefficient de régression (B) proche de un**

Cette configuration indique que le réseau de neurones est le seul capable d'estimer avec précision la probabilité réelle de défaut.

---

## 6. Applications Pratiques

### 6.1 Gestion des Risques

Ce dataset est particulièrement utile pour :
- Développer des modèles de scoring de crédit
- Évaluer le risque de défaut des clients
- Optimiser les décisions d'octroi de crédit
- Améliorer les stratégies de recouvrement

### 6.2 Cas d'Usage

- **Institutions financières** : Évaluation du risque crédit
- **Recherche académique** : Comparaison d'algorithmes de classification
- **Data Scientists** : Projets d'apprentissage automatique
- **Analystes financiers** : Modélisation prédictive

---

## 7. Accès aux Données

### 7.1 Installation via Python

```python
pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

# Télécharger le dataset
default_of_credit_card_clients = fetch_ucirepo(id=350)

# Données (en tant que pandas dataframes)
X = default_of_credit_card_clients.data.features
y = default_of_credit_card_clients.data.targets

# Métadonnées
print(default_of_credit_card_clients.metadata)

# Informations sur les variables
print(default_of_credit_card_clients.variables)
```

### 7.2 Téléchargement Direct

Le dataset peut également être téléchargé directement au format Excel (.xls) - taille : 5.3 MB

---

## 8. Publications Associées

### 8.1 Article Principal

**"The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients"**
- Auteurs : I-Cheng Yeh, Che-hui Lien
- Année : 2009
- Publié dans : Expert Systems with Applications

### 8.2 Articles Citant ce Dataset

1. **"Credit Default Mining Using Combined Machine Learning and Heuristic Approach"** (2018)
   - Auteurs : Sheikh Islam, William Eberle, Sheikh Ghafoor
   
2. **"Dancing in the Dark: Private Multi-Party Machine Learning in an Untrusted Setting"** (2018)
   - Auteurs : Clement Fung, Jamie Koerner, Stewart Grant, Ivan Beschastnikh

---

## 9. Recommandations

### 9.1 Pour les Praticiens

- Utiliser ce dataset comme benchmark pour comparer différents algorithmes de classification
- Explorer les techniques d'ensemble pour améliorer les performances
- Considérer l'importance des variables temporelles (historique de paiement)
- Implémenter des mécanismes de validation croisée robustes

### 9.2 Pour les Chercheurs

- Approfondir l'analyse de la méthode "Sorting Smoothing Method"
- Explorer d'autres architectures de réseaux de neurones
- Étudier l'impact de l'ingénierie des features
- Comparer avec des méthodes plus récentes (XGBoost, Deep Learning, etc.)

---

## 10. Limitations et Considérations

### 10.1 Limitations du Dataset

- Données spécifiques à Taiwan (généralisation à d'autres marchés à valider)
- Période temporelle limitée (avril-septembre 2005)
- Pas de données sur les facteurs macroéconomiques
- Informations limitées sur le contexte individuel des clients

### 10.2 Considérations Éthiques

- Protection des données personnelles
- Équité dans l'octroi de crédit
- Risque de biais discriminatoires
- Transparence des modèles de décision
<img src="graphe 1.png" style="height:200px;margin-right:150px"/>
<img src="graphe 2:png" style="height:200px;margin-right:150px"/>
---

## 11. Conclusion

Le dataset "Default of Credit Card Clients" constitue une ressource précieuse pour le développement et l'évaluation de modèles prédictifs de défaut de paiement. Avec 30 000 instances et 23 variables bien documentées, il offre un terrain d'expérimentation robuste pour les chercheurs et praticiens en data science et risk management.

La recherche originale démontre la supériorité des réseaux de neurones artificiels pour estimer la probabilité de défaut, ouvrant la voie à des applications pratiques dans le secteur bancaire et financier.

---

## 12. Références

- **Dataset Citation:** Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- **URL:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

*Rapport généré le 12 novembre 2025




