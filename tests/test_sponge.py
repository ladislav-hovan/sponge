import pytest

def please_dont_execute_this():
    assert False

### Unit tests ###

# Helper functions
import sponge.helper_functions as helper_f

@pytest.mark.parametrize("input, expected_output", [
    (0, 0),
    (0.5, -0.5),
    (1, 0),
])
def test_plogp(input, expected_output):
    assert helper_f.plogp(input) == expected_output


from sponge.test_motifs import *

def test_calculate_ic_no_info(no_info_motif):
    assert helper_f.calculate_ic(no_info_motif) == 0


def test_calculate_ic_all_the_same(all_A_motif):
    # Length of the test motif is 6, so expected value is 2 * 6 = 12
    assert helper_f.calculate_ic(all_A_motif) == 12


def test_calculate_ic_SOX2(SOX2_motif):
    assert (helper_f.calculate_ic(SOX2_motif) == 
        pytest.approx(12.95, abs=0.01))


@pytest.mark.parametrize("input, expected_output", [
    ('CAB', 'Cab'),
    ('SOX2', 'SOx2'),
    ('ARHGAP21', 'ARHGAP21'),
    ('ABC2DE', 'ABC2de'),
])
def test_adjust_gene_name(input, expected_output):
    assert helper_f.adjust_gene_name(input) == expected_output


import datetime

@pytest.mark.parametrize("input, expected_output", [
    ('a_string', 'a_string'),
    (datetime.datetime(1992, 5, 29, 23, 15), '29/05/1992, 23:15:00')
])
def test_parse_datetime(input, expected_output):
    assert helper_f.parse_datetime(input) == expected_output


# Filtering functions
import sponge.filtering as filter_f 


# File retrieval functions
import sponge.file_retrieval as file_f


### Integration tests ###
import os

from sponge.sponge import Sponge

from shutil import rmtree

# def test_show_tmp_path(tmp_path):
#     print (tmp_path)
#     print (f'{tmp_path}/ppi_prior.tsv')
#     assert False


@pytest.mark.skip
def test_full_default_workflow():
    temp_folder = '.sponge_temp/'
    ppi_output = 'ppi_prior.tsv'
    motif_output = 'motif_prior.tsv'

    if os.path.exists(temp_folder):
        rmtree(temp_folder)
    for file in [ppi_output, motif_output]:
        if os.path.exists(file):
            os.remove(file)

    sponge_obj = Sponge(
        run_default=True,
        temp_folder=temp_folder,
        ppi_outfile=ppi_output,
        motif_outfile=motif_output,
    )
    sponge_obj.clear_cache()

    assert os.path.exists(ppi_output)
    assert os.path.exists(motif_output)