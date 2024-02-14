### Imports ###
import pandas as pd

from math import log2

from Bio.motifs.jaspar import Motif

from typing import Union

from datetime import datetime

### Functions ###
def plogp(
    x: float,
) -> float:   
    """
    Returns x*log2(x) for a number, handles the 0 case properly.

    Parameters
    ----------
    x : float
        The input value

    Returns
    -------
    float
        The value of x*log2(x)
    """

    if x == 0:
        return 0
    else:
        return x*log2(x)


def calculate_ic(
    motif: Motif,
) -> float:
    """
    Calculates the information content for a given motif, assuming equal 
    ACGT distribution.

    Parameters
    ----------
    motif : Motif
        A JASPAR Motif object

    Returns
    -------
    float
        The information content of the motif
    """

    df = pd.DataFrame(motif.pwm)
    # Calculate the IC for each position in the motif
    df['IC'] = df.apply(lambda x: 2 + sum([plogp(x[y]) for y in 
        ['A', 'C', 'G', 'T']]), axis=1)
    
    # Return the total IC for the whole motif
    return df['IC'].sum()


def adjust_gene_name(
    gene: str,
) -> str:
    """
    Adjusts the gene name by converting the last two letters to 
    lowercase. This is typically done to enhance name matching.

    Parameters
    ----------
    gene : str
        The provided gene name

    Returns
    -------
    str
        The adjusted gene name
    """
    
    return gene[:-2] + gene[-2:].lower()


def parse_datetime(
    datetime: Union[str, datetime],
) -> str:
    """
    Converts the provided datetime object into a formatted string, 
    or returns the provided string. 

    Parameters
    ----------
    datetime : Union[str, datetime]
        The provided string or datetime object

    Returns
    -------
    str
        The provided string or provided datetime expressed as string
    """
    
    if type(datetime) == str:
        return datetime
    else:
        return datetime.strftime('%d/%m/%Y, %H:%M:%S')