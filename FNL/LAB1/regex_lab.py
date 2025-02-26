import os
import re
import math
import unittest

class RegexExercises:
    """
    Exercices d'expressions régulières

    Implémentez les méthodes suivantes pour que les tests unitaires passent.
    Ces méthodes renvoient une chaîne littérale contenant une expression
    régulière satisfaisant la condition décrite dans la documentation.
    """

    
    def planets():
        """
        Correspond aux noms des huit planètes, en majuscule initiale (c'est-à-dire,
        Mercury, Venus, Earth, etc.). Il y a 44 lignes dans BleakHouse.txt qui
        correspondent à cette expression.
        """
        return r"Earth|Jupiter|Neptune|Mars|Saturn|Venus|Uranus|Mercury"  # cette première expression est déjà réalisée pour vous

    
    def dashes():
        """
        Correspond à deux tirets consécutifs: --. Il y a 1802 lignes dans BleakHouse.txt
        qui comportent deux tirets.
        """
        return r"--" # A compléter

    
    def quotes():
        """
        Correspond à une chaîne entre guillemets, c'est-à-dire une chaîne qui commence
        et finit par des guillemets doubles sur une seule ligne et qui contient
        au moins un caractère entre les guillemets. Pour être certain qu'il ne s'agit
        pas de la fin d'une citation et du début d'une autre, assurez-vous que le premier
        guillemet soit au début d'une ligne ou que le caractère avant lui soit un caractère
        non alphanumérique, et que le deuxième guillemet ait un caractère non alphanumérique
        après lui.
        """

        return r"(^\"|[^\w]\")[^\"\n]+\"[^\w]"

    
    def rain():
        """
        Correspond à un match contenant des mots qui commencent par "rain". Cela inclut
        "Rain", "raindrop", "raining", etc., mais pas "brains". 
        """
        return r"(^[rR]ain[A-Za-z]*)[^A-Za-z]*"

    
    def east():
        """
        Correspond à un match avec le mot "east". Cela inclut "East" mais pas "eastern".
        """
        return ""  # bordure de mot pour ne pas inclure "eastern"

    
    def hyphenated():
        """
        Correspond à un match avec des mots comportant un tiret. Par exemple, cela inclut
        "assembly-room", "half-a-dozen" et même "en-r-r-r-raged."
        """
        return ""  # un mot suivi d'un tiret et d'au moins un mot

    # ----------------------------------------------------------------------
    # Maintenant, nous quittons Bleak House pour étudier les nombres. Chaque ligne du 
    # fichier numbers.txt contient un nombre possible, suivi d'espaces, 
    # suivi d'un commentaire. Écrivez donc des expressions régulières qui 
    # correspondent à des chaînes commençant au début de la ligne et suivies 
    # d'espaces. Attention, les caractères d'espaces doivent être inclus dans
    # votre expression régulière (càd qu'elle finira par \s+)
    # ----------------------------------------------------------------------

    
    def digits():
        """
        Correspond à une séquence de chiffres. Cela doit correspondre seulement à
        la première ligne de numbers.txt.
        """
        return ""  # ^ pour le début de ligne, puis chiffres, puis espaces

    
    def ssn():
        """
        Correspond aux numéros de sécurité sociale. Cela doit correspondre seulement
        à la deuxième ligne de numbers.txt.
        """
        return ""  # format type ssn suivi d'espaces

    
    def commaNumbers():
        """
        Correspond aux nombres avec des virgules, comme 23,354.
        Ce motif doit correspondre uniquement aux lignes troisième, quatrième, cinquième
        et sixième du fichier numbers.txt.
        """
        return ""  # un ou plusieurs groupes de chiffres séparés par des virgules

    
    def decimalNumbers():
        """
        Correspond à une chaîne de chiffres avec un point décimal.
        Le point décimal peut se trouver en début ou en fin, mais il doit être présent,
        et il doit y avoir au moins un chiffre. Ce motif doit correspondre uniquement aux
        neuvième, dixième et onzième lignes de numbers.txt.
        """
        # On autorise soit: chiffre(s)+point ou point+chiffre(s) ou chiffre(s)+point+chiffres
        return ""

    
    def realNumbers():
        """
        Correspond à des nombres réels qui commencent éventuellement par un signe (+ ou -),
        suivis d'une séquence d'au moins un chiffre qui peut contenir un point, éventuellement
        suivi par un "e" ou "E", suivi d'un signe optionnel et d'au moins un chiffre.
        Ce motif doit correspondre à la première, aux neuvième à onzième, et aux cinq dernières lignes
        du fichier numbers.txt (c'est-à-dire, toutes les lignes contenant des nombres sans virgules).
        """
        return ""


