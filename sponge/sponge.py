### Imports ###
import pickle

import numpy as np

from datetime import datetime

from biomart import BiomartServer

from collections import defaultdict

from pyjaspar import jaspardb

from typing import Mapping

from multiprocessing import Pool

from math import ceil, sqrt

from sponge.functions import *

from shutil import rmtree

### Class definition ###
class Sponge:
    """
    Sponge class can process the data necessary for creating a prior
    TF-gene regulatory network, along with a prior protein-protein 
    interaction network. It also contains tools to download these data,
    so that minimal input from the user is required. The networks are 
    provided in a format compatible with PANDA/LIONESS and other
    NetZoo tools.

    The run_default option is implemented in the constructor which will
    run the entire pipeline after class creation using the provided
    options and defaults where appropriate.

    Usage:
    sponge_obj = Sponge(run_default=True)
    OR
    sponge_obj = Sponge()
    sponge_obj.select_tfs()
    sponge_obj.find_human_homologs()
    sponge_obj.filter_matches()
    sponge_obj.retrieve_ppi()
    sponge_obj.write_ppi_prior()
    sponge_obj.aggregate_matches()
    sponge_obj.write_motif_prior()

    Most functions have options which can usually also be provided 
    to the constructor, for more details refer to the documentation of 
    the individual functions.
    """

    def __init__(
        self,
        temp_folder: Union[str, bytes, os.PathLike] = '.sponge_temp',
        run_default: bool = False,
        jaspar_release: Optional[str] = None,
        n_processes: int = 1,
        paths_to_files: Mapping[str, Union[str, bytes, os.PathLike]] = {},
        tss_offset: Tuple[int, int] = (-750, 250),
        prompt: bool = True
    ):
        """
        Initialises an instance of the Sponge class.

        Parameters
        ----------
        temp_folder : Union[str, bytes, os.PathLike], optional
            The temporary folder for saving downloaded files, 
            by default '.sponge_temp'
        run_default : bool, optional
            Whether to run the default pipeline automatically after
            class creation, by default False
        """

        self.temp_folder = temp_folder
        self.ensembl = None
        self.ppi_frame = None
        self.motif_frame = None
        self.n_processes = n_processes
        self.fingerprint = defaultdict(dict)
        self.provided_paths = paths_to_files
        self.tss_offset = tss_offset

        # JASPAR initialisation is done here to allow for straightforward
        # early termination in case of unrecognised JASPAR release
        # TODO: Move it to a function
        self.jdb_obj = jaspardb()
        if jaspar_release is None:
            from pyjaspar import JASPAR_LATEST_RELEASE
            self.jaspar_release = JASPAR_LATEST_RELEASE
        else:
            jaspar_available = self.jdb_obj.get_releases()
            if jaspar_release in jaspar_available:
                self.jaspar_release = jaspar_release
                self.jdb_obj = jaspardb(self.jaspar_release)
            elif 'JASPAR' + jaspar_release in jaspar_available:
                print (f'Found {"JASPAR" + jaspar_release} in available '
                    'releases, assuming this matches your choice')
                self.jaspar_release = 'JASPAR' + jaspar_release
                self.jdb_obj = jaspardb(self.jaspar_release)
            else:
                print ('The specified version of the JASPAR release '
                    f'({jaspar_release}) is not available')
                print ('Available versions:')
                print (', '.join(jaspar_available))
                print ('The downstream pipeline will not be run')
                return
        self.log_fingerprint('JASPAR', self.jaspar_release)

        self.files_ready = self.prepare_files(prompt)

        if run_default and self.files_ready:
            self.run_default_workflow()


    ### File retrieval functions ###
    def prepare_files(
        self,
        prompt: bool = True
    ) -> bool:      

        print ()
        print ('--- Running prepare_files() ---')

        provided = []
        for k,v in self.provided_paths.items():
            if k not in FILE_DF.index:
                print (f'Unrecognised file type: {k}, ignoring provided path')
                continue
            if not os.path.exists(v):
                print (f'The path provided for file type {k} ({v}) is invalid '
                    'and will be ignored')
            else:
                provided.append(k)
                self.log_fingerprint(k.upper(), '', provided=True)
        self.provided_paths = {k: v for k,v in self.provided_paths.items()
            if k in provided}

        to_check = [file for file in FILE_DF.index if file not in provided]
        to_retrieve = {}
        for file in to_check:
            if not self.check_file_exists(file):
                to_retrieve[file] = FILE_DF.loc[file, 'name']
            else:
                self.log_fingerprint(file.upper(), '', cached=True)
        
        if len(to_retrieve) == 0:
            print ('All the required files were provided or located in the '
                f'{self.temp_folder} directory')
            return True
        else:
            print ('The following files were not found:')
            print (', '.join(to_retrieve.values()))
            if prompt:
                reply = self.prompt_to_confirm('Do you want to download them '
                    'automatically?')
            else:
                print ('They will be downloaded automatically')
                print ()
            if not prompt or reply:
                for description in to_retrieve.keys():
                    file_path = self.retrieve_file(description, prompt=False)
                    # This shouldn't really happen as we're only using valid
                    # descriptions, but better safe than sorry
                    if file_path is None:
                        print (f'File {to_retrieve[description]} could not be '
                            'retrieved')
                        return False
            else:
                return False
            

    def prompt_to_confirm(
        self,
        question: str
    ) -> bool:
        
        key = None
        positive = ['y', 'yes', 'hell yeah']
        negative = ['n', 'no', 'nope']
        while key is None or key.lower() not in positive + negative:
            if key is not None:
                print (f'Input not recognised: {key}')
            print (f'{question} Y/N', flush=True)
            key = input()
            print (key)
        print ()

        return key.lower() in positive


    def description_to_path(
        self,
        description: str
    ) -> Optional[str]:

        if description not in FILE_DF.index:
            print (f'File description not recognised: {description}')
            return None
        file_name = FILE_DF.loc[description, 'name']
        file_path = os.path.join(self.temp_folder, file_name)

        return file_path


    def check_file_exists(
        self,
        description: str
    ) -> bool:
        
        return os.path.exists(self.description_to_path(description))


    def retrieve_file(
        self,
        description: str,
        prompt: bool = True
    ) -> Optional[str]:
        
        if description in self.provided_paths:
            print ('Using provided file', self.provided_paths[description])
            print ()
            return self.provided_paths[description]
        file_path = self.description_to_path(description)
        if not os.path.exists(self.temp_folder):
            os.mkdir(self.temp_folder)
        if os.path.exists(file_path):
            print ('Using cached file', file_path)
            print ()
        else:
            print (f'File {FILE_DF.loc[description, "name"]} not found ' 
                f'in directory {self.temp_folder}')
            if prompt:
                reply = self.prompt_to_confirm('Do you want to download it?')
                if not reply:
                    return None
            to_request = FILE_DF.loc[description, 'url']
            if to_request is None:
                # These options are not unused: they are passed to the
                # evaluated function call as kwargs
                options = {'file_path': file_path}
                if description == 'promoter':
                    options['tss_offset'] = self.tss_offset
                eval(FILE_DF.loc[description, 'eval'])
            else:
                if description == 'jaspar_bigbed':
                    if self.jaspar_release is None:
                        raise ValueError('The release of jaspar has to be '
                            'specified in order to retrieve the bigbed file')
                    to_request = to_request.format(
                        year=self.jaspar_release[-4:])
                    version = self.jaspar_release
                elif description == 'homologene':
                    version_url = '/'.join(to_request.split('/')[:-1] + 
                        ['RELEASE_NUMBER'])
                    version_request = requests.get(version_url, stream=True)
                    version = version_request.text.strip()
                print (f'Downloading data into {file_path}...')
                download_with_progress(to_request, file_path)
                print ()
                self.log_fingerprint(description.upper(), version)

        return file_path


    def load_promoters_from_biomart(
        self,
        file_path: Union[str, bytes, os.PathLike],
        filter_basic: bool = True,
        chromosomes: Iterable[str] = [str(i) for i in range(1,23)] + 
            ['MT', 'X', 'Y'],
        tss_offset: Tuple[int, int] = (-750, 250),
        keep_ensembl: bool = True
    ) -> None:

        bm_server = BiomartServer(ENSEMBL_URL)
        ensembl = bm_server.datasets['hsapiens_gene_ensembl']
        attributes = ['ensembl_transcript_id', 'transcript_gencode_basic', 
            'chromosome_name', 'transcription_start_site', 'strand']
        if keep_ensembl:
            attributes += ['ensembl_gene_id', 'external_gene_name', 
                'gene_biotype']
        print ('Retrieving response to query...')
        response = ensembl.search({'attributes': attributes}, header=1)
        buffer = download_with_progress(response)
        self.log_fingerprint('PROMOTER', ensembl.display_name)
        dtype_dict = defaultdict(lambda: str)
        dtype_dict['Transcription start site (TSS)'] = int
        dtype_dict['Strand'] = int
        df = pd.read_csv(buffer, sep='\t', dtype=dtype_dict)
        print ('Filtering and modifying dataframe...')
        if filter_basic:
            df = df[df['GENCODE basic annotation'] == 'GENCODE basic'].copy()
            df.drop(columns='GENCODE basic annotation', inplace=True)
        if chromosomes is not None:
            df = df[df['Chromosome/scaffold name'].isin(chromosomes)]
        df['Chromosome'] = df['Chromosome/scaffold name'].apply(lambda x: 
            'chrM' if x == 'MT' else f'chr{x}')
        df['Strand'] = df['Strand'].apply(lambda x: '+' if x > 0 else '-')
        df['Start'] = df.apply(lambda row: 
            row['Transcription start site (TSS)'] + tss_offset[0] 
            if row['Strand'] == '+' 
            else row['Transcription start site (TSS)'] - tss_offset[1], 
            axis=1)
        df['End'] = df['Start'] + (tss_offset[1] - tss_offset[0])
        df['Score'] = 0
        df.sort_values(['Chromosome', 'Start'], inplace=True)
        columns = ['Chromosome', 'Start', 'End', 'Transcript stable ID', 
            'Score', 'Strand']
        print (f'Saving data to {file_path}...')
        df[columns].to_csv(file_path, sep='\t', header=False, index=False)
        print ()
        if keep_ensembl:
            self.ensembl = df[['Gene stable ID', 'Transcript stable ID', 
                'Gene name', 'Gene type']]


    def load_ensembl_from_biomart(
        self,
        file_path: Union[str, bytes, os.PathLike]
    ) -> None:
        
        bm_server = BiomartServer(ENSEMBL_URL)
        ensembl = bm_server.datasets['hsapiens_gene_ensembl']
        attributes = ['ensembl_transcript_id', 'ensembl_gene_id', 
            'external_gene_name', 'gene_biotype']
        print ('Retrieving response to query...')
        response = ensembl.search({'attributes': attributes}, header=1)
        buffer = download_with_progress(response)
        self.log_fingerprint('ENSEMBL', ensembl.display_name)
        df = pd.read_csv(buffer, sep='\t')     
        df.to_csv(file_path, sep='\t', index=False)
        self.ensembl = df


    def log_fingerprint(
        self,
        label: str,
        version: str,
        provided: bool = False,
        cached: bool = False
    ) -> None:
        
        self.fingerprint[label]['version'] = version
        self.fingerprint[label]['datetime'] = datetime.fromtimestamp(
            time.time())
        self.fingerprint[label]['provided'] = provided
        self.fingerprint[label]['cached'] = cached

        fingerprint_file = os.path.join(self.temp_folder, '.fingerprint')
        if os.path.exists(fingerprint_file):
            temp_fingerprint = pickle.load(open(fingerprint_file, 'rb'))
        else:
            temp_fingerprint = {}
        if not provided and not cached:
            temp_fingerprint[label] = self.fingerprint[label]
            pickle.dump(temp_fingerprint, open(fingerprint_file, 'wb'))
        if cached:
            self.fingerprint[label] = temp_fingerprint[label]
            self.fingerprint[label]['cached'] = True


    ### Main workflow functions ###
    def run_default_workflow(  
        self
    ) -> None:

        self.select_tfs()
        self.find_human_homologs(prompt=False)
        self.filter_matches(prompt=False)
        self.retrieve_ppi()
        self.write_ppi_prior()
        self.aggregate_matches(prompt=False)
        self.write_motif_prior()


    def select_tfs(
        self,
        drop_heterodimers: bool = True
    ) -> None:
        """
        Selects transcription factors from the newest version of the
        JASPAR database and stores them in the class instance.

        Parameters
        ----------
        drop_heterodimers : bool, optional
            Whether to drop filter out heterodimers, by default True
        """

        print ()
        print ('--- Running select_tfs() ---')
        print (f'Using: {self.jaspar_release}')

        # All vertebrate motifs
        motifs = self.jdb_obj.fetch_motifs(collection='CORE', 
            tax_group='vertebrates', all_versions=True)
        print ()
        print ('All motif versions:', len(motifs))
        print ('Motif base IDs:', len(set([i.base_id for i in motifs])))

        # Select latest, preferring human ones
        latest = {}
        for i in motifs:
            if i.base_id not in latest:
                latest[i.base_id] = [i.matrix_id, i.species]
            else:
                # Replace with newer version if the new one is human or the old 
                # one isn't       
                if (('9606' in i.species) or 
                    ('9606' not in latest[i.base_id][1])):
                    # This could be added to the logical condition above but 
                    # this is more readable
                    if int(i.matrix_id[-1]) > int(latest[i.base_id][0][-1]):
                        latest[i.base_id] = [i.matrix_id, i.species]
        motifs_latest = [i for i in motifs if 
            i.matrix_id == latest[i.base_id][0]]
        
        # Keep only one motif per TF
        # Consider dropping this requirement maybe
        tf_to_motif = {}
        for i in motifs_latest:
            if i.name not in tf_to_motif:
                tf_to_motif[i.name] = {i.matrix_id: calculate_ic(i)}
            else:
                tf_to_motif[i.name][i.matrix_id] = calculate_ic(i)
        self.tf_to_motif = tf_to_motif
        motifs_unique = [i for i in motifs_latest if 
            (tf_to_motif[i.name][i.matrix_id] == 
            max(tf_to_motif[i.name].values()))]
        print ('Unique motifs:', len(motifs_unique))

        # Drop heterodimers
        if drop_heterodimers:
            motifs_nohd = [i for i in motifs_unique if '::' not in i.name]
            print ('Motifs without heterodimers:', len(motifs_nohd))
            self.motifs = motifs_nohd
        else:
            self.motifs = motifs_unique


    def find_human_homologs(
        self, 
        homologene_file: Optional[Union[str, bytes, os.PathLike]] = None,
        prompt: bool = True
    ) -> None:
        """
        Attempts to map all initially selected non-human transcription
        factors to their human homologs.

        Parameters
        ----------
        homologene_file : Optional[Union[str, bytes, os.PathLike]], 
            optional
            The path to a homologene file or None to use cache or
            download it, by default None
        prompt : bool, optional
            Whether to prompt before downloading, by default True
        """
        
        print ()
        print ('--- Running find_human_homologs() ---')

        if homologene_file is None:
            homologene_file = self.retrieve_file('homologene', prompt=prompt)
            if homologene_file is None:
                print ('Unable to find or retrieve the homologene file, ' 
                    'exiting')
                return

        # Get the non-human motifs
        non_human_motifs = [i for i in self.motifs if '9606' not in i.species]
        print ('Non-human motifs:', len(non_human_motifs))

        # Read the homologene database
        hg_df = pd.read_csv(homologene_file, sep='\t', header=None, 
                    names=['HG Group ID', 'TaxID', 'Gene ID', 
                           'Gene Symbol', 'Protein GI', 'Protein Accession'])

        # Get the non-human motif names
        non_human_motif_names = [i.name for i in non_human_motifs]
        # Compare against homologene
        found_names = hg_df[hg_df['Gene Symbol'].isin([adjust_gene_name(i) for 
            i in non_human_motif_names])]['Gene Symbol'].unique()
        # Find the missing ones
        missing = (set([adjust_gene_name(i) for i in non_human_motif_names]) - 
            set(found_names))
        print ()
        print ('Names missing from the homologene database:')
        for i in [(i.name,i.acc) for i in non_human_motifs if 
            i.name in missing]:
            print (i[0], *i[1])

        # Get the missing IDs from Uniprot API
        print ()
        print ('Retrieving matches from UniProt...')
        mapping = get_uniprot_mapping("UniProtKB_AC-ID", "RefSeq_Protein",
            [i.acc[0] for i in non_human_motifs if i.name in missing])
        mapping.columns = ['Uniprot', 'Accession']

        # Create a DataFrame for matching missing entries
        hg_df[hg_df['Protein Accession'].isin(mapping['Accession'])]
        missing_df = pd.DataFrame([(i.name, i.acc[0]) for i in non_human_motifs 
            if i.name in missing], columns = ['Gene Symbol', 'Uniprot'])        
        matching_df = missing_df.join(mapping.set_index('Uniprot'), 
            on='Uniprot').join(hg_df.set_index('Protein Accession'), 
            on='Accession', rsuffix='_HG')

        def corresponding_id(name):
            values = hg_df[hg_df['Gene Symbol'] == name]['HG Group ID'].values
            if len(values) == 0:
                matches = matching_df[matching_df['Gene Symbol'] == name]
                values = matches['HG Group ID'].values
            return values

        # Create a DataFrame of corresponding names
        corr_df = pd.DataFrame(non_human_motif_names, 
            columns=['Original Name'])
        corr_df['Adjusted Name'] = corr_df['Original Name'].apply(
            adjust_gene_name)
        corr_df['Group ID'] = corr_df['Adjusted Name'].apply(corresponding_id)
        corr_df['Group ID'] = corr_df['Group ID'].apply(lambda x: 
            x[0] if len(x) > 0 else np.nan)
        corr_df['Human Name'] = corr_df['Group ID'].apply(lambda x: 
            hg_df[(hg_df['HG Group ID'] == x) & 
            (hg_df['TaxID'] == 9606)]['Gene Symbol'].values)
        corr_df['Human Name'] = corr_df['Human Name'].apply(lambda x: 
            x[0] if len(x) > 0 else '')
        corr_df['Trivial'] = corr_df['Original Name'].apply(lambda x: 
            x.upper()) == corr_df['Human Name']

        # Find duplicates
        duplicated = corr_df[corr_df['Human Name'].duplicated(keep=False) & 
            (corr_df['Human Name'] != '')].copy()
        to_print = duplicated.groupby('Human Name')['Original Name'].unique(
            ).apply(lambda x: ' '.join(x))
        print ()
        print ('Duplicate names:')
        for i in to_print.index:
            print (f'{i}:', to_print.loc[i])

        # Calculate the information content for duplicates
        duplicated['IC'] = duplicated['Original Name'].apply(lambda x: 
            max(self.tf_to_motif[x].values()))
        # Keep the highest IC amongst the duplicates
        to_drop = duplicated['Original Name'][duplicated.sort_values(
            'IC').duplicated('Human Name', keep='last')]

        # Exlude the IDs which are already present among the human ones
        human_motif_names = [i.name for i in self.motifs if 
            '9606' in i.species]
        corr_df['Duplicate'] = corr_df['Human Name'].isin(human_motif_names)

        # Perform the final filtering - discard all duplicates and TFs without 
        # homologs
        corr_df_final = corr_df[(corr_df['Duplicate'] == False) & 
            (corr_df['Human Name'] != '') & 
            (corr_df['Original Name'].isin(to_drop) == False)]

        # The mapping of original to human names and the matrix IDs to be kept
        animal_to_human = {animal_name: human_name for animal_name, human_name 
            in zip(corr_df_final['Original Name'], 
            corr_df_final['Human Name'])}
        print ()
        print ('Final number of IDs which will be replaced by human homologs:', 
               len(animal_to_human))
        matrix_ids = [motif.matrix_id for motif in self.motifs if 
            (motif.name in human_motif_names or motif.name in animal_to_human)]
        print ('Final number of total matrix IDs:', len(matrix_ids))

        self.animal_to_human = animal_to_human
        self.matrix_ids = matrix_ids
        self.tf_names = human_motif_names + list(animal_to_human.keys())


    def filter_matches(
        self, 
        promoter_file: Optional[Union[str, bytes, os.PathLike]] = None, 
        bigbed_file: Optional[Union[str, bytes, os.PathLike]] = None,
        score_threshold: float = 400,
        chromosomes: Iterable[str] = [f'chr{i}' for i in [j for j in 
            range(1, 23)] + ['M', 'X', 'Y']],
        n_processes: Optional[int] = None,
        prompt: bool = True
    ) -> None:
        
        print ()
        print ('--- Running filter_matches() ---')

        if n_processes is None:
            n_processes = self.n_processes

        if promoter_file is None:
            promoter_file = self.retrieve_file('promoter', prompt=prompt)
            if promoter_file is None:
                print ('Unable to find or retrieve the promoter file, exiting')
                return
        
        if bigbed_file is None:
            bigbed_file = self.retrieve_file('jaspar_bigbed', prompt=prompt)
            if bigbed_file is None:
                print ('Unable to find or retrieve the JASPAR bigbed file, '
                    'exiting')
                return

        print ('Loading the promoter bed file...')
        df_full = bioframe.read_table(promoter_file, schema='bed')
        df_full['name'] = df_full['name'].apply(lambda x: x.split('.')[0])
        df_full.drop(columns=['score', 'strand'], inplace=True)
        df_full.set_index('name', inplace=True)

        results_list = []
        p = Pool(n_processes)

        print ()
        print ('Iterating over the chromosomes...')
        start_time = time.time()
        for chrom in chromosomes:
            st_chr = time.time()
            df_chrom = df_full[df_full['chrom'] == chrom]
            if len(df_chrom) == 0:
                suffix = 'no transcripts'
            elif len(df_chrom) == 1:
                suffix = '1 transcript'
            else:
                suffix = f'{len(df_chrom)} transcripts'
            print (f'Chromosome {chrom[3:]} with ' + suffix)
            chunk_size = ceil(sqrt(len(df_chrom) / n_processes))
            chunk_divisions = [i for i in range(0, len(df_chrom), chunk_size)]
            input_tuples = [(bigbed_file, df_chrom, self.tf_names, chrom, i, 
                i+chunk_size, score_threshold) for i in chunk_divisions]
            result = p.map_async(filter_edges_helper, input_tuples, 
                chunksize=n_processes)
            edges_chrom_list = result.get()
            results_list += edges_chrom_list
            elapsed_chr = time.time() - st_chr
            print (f'Done in: {elapsed_chr // 60:n} m '
                f'{elapsed_chr % 60:.2f} s')

        elapsed = time.time() - start_time
        print ()
        print (f'Total time: {elapsed // 60:n} m {elapsed % 60:.2f} s')

        self.all_edges = pd.concat(results_list, ignore_index=True)

    
    def load_matches(
        self,
        file_path: Union[str, bytes, os.PathLike]
    ):
        """
        Loads the filtered matches from a file, allows the use of
        the downstream SPONGE functions without running the steps up to 
        and including filter_matches 

        Parameters
        ----------
        file_path : Union[str, bytes, os.PathLike]
            The path to a file that contains the filtered matches 
            in a format compatible with what filter_matches generates
        """
        
        self.all_edges = pd.read_csv(file_path, sep='\t')


    def retrieve_ppi(
        self
    ) -> None:
        
        print ()
        print ('--- Running retrieve_ppi() ---')
        
        filtered_tfs = self.all_edges['TFName'].unique()
        humanised_tfs = [self.animal_to_human[x] if x in self.animal_to_human 
            else x for x in filtered_tfs]
        query_string = '%0d'.join(humanised_tfs)

        print ('Retrieving mapping from STRING...')
        mapping_request = requests.get(f'{STRING_URL}get_string_ids?'
            f'identifiers={query_string}&species=9606')
        mapping_df = pd.read_csv(BytesIO(mapping_request.content), sep='\t')
        mapping_df['queryName'] = mapping_df['queryIndex'].apply(
            lambda i: humanised_tfs[i])
        diff_df = mapping_df[mapping_df['queryName'] != 
            mapping_df['preferredName']]
        ids_to_check = np.concatenate((diff_df['queryName'], 
            diff_df['preferredName']))
        version_request = requests.get(f'{STRING_URL}version')
        version_df = pd.read_csv(BytesIO(version_request.content), sep='\t')
        self.log_fingerprint('STRING', version_df['string_version'])
        
        print ('Checking the conflicts in the UniProt database...')
        uniprot_df = get_uniprot_mapping('Gene_Name', 'UniProtKB', 
            ids_to_check).set_index('from')
        matching_ids = list(mapping_df[mapping_df['queryName'] == 
            mapping_df['preferredName']]['preferredName'])
        p_to_q = {p: q for q,p in zip(diff_df['queryName'], 
            diff_df['preferredName'])}
        for p,q in p_to_q.items():
            if (p not in uniprot_df.index or q not in uniprot_df.index
                or uniprot_df.loc[p, 'to'] == uniprot_df.loc[q, 'to']):
                matching_ids.append(p)
        query_string_filt = '%0d'.join(matching_ids)

        print ('Retrieving the network from STRING...')
        request = requests.get(f'{STRING_URL}network?'
            f'identifiers={query_string_filt}&species=9606')
        ppi_df = pd.read_csv(BytesIO(request.content), sep='\t')

        print ('Processing the results...')
        ppi_df.drop(['stringId_A', 'stringId_B', 'ncbiTaxonId', 'nscore', 
            'fscore', 'pscore', 'ascore', 'escore', 'dscore', 'tscore'], 
            axis=1, inplace=True)
        ppi_df.rename(columns={'preferredName_A': 'tf1', 
            'preferredName_B': 'tf2'}, inplace=True)
        ppi_df['tf1'].replace(p_to_q, inplace=True)
        ppi_df['tf2'].replace(p_to_q, inplace=True)
        ppi_df.sort_values(by=['tf1', 'tf2'], inplace=True)

        print ()
        print ('Final number of TFs in the PPI network: '
            f'{len(set(ppi_df["tf1"]).union(set(ppi_df["tf2"])))}')
        print (f'Final number of edges: {len(ppi_df)}')

        self.ppi_frame = ppi_df


    def write_ppi_prior(
        self,
        output_path: Union[str, bytes, os.PathLike] = 'ppi_prior.tsv',
        weighted: bool = False
    ) -> None:
        
        print ()
        print ('--- Running write_ppi_prior() ---')
        
        if self.ppi_frame is None:
            print ('No motif prior has been generated yet, please run '
                'retrieve_ppi() first')
            return
        
        if weighted:
            self.ppi_frame[['tf1', 'tf2', 'score']].to_csv(output_path, 
                sep='\t', index=False, header=False)
        else:
            self.ppi_frame['edge'] = 1
            self.ppi_frame[['tf1', 'tf2', 'edge']].to_csv(output_path, 
                sep='\t', index=False, header=False)
            

    def aggregate_matches(
        self,
        ensembl_file: Optional[Union[str, bytes, os.PathLike]] = None,
        prompt: bool = True,
        use_gene_names: bool = True,
        protein_coding_only: bool = False
    ) -> None:
        
        print ()
        print ('--- Running aggregate_matches() ---')

        if self.ensembl is None or ensembl_file is not None:
            if ensembl_file is None:
                ensembl_file = self.retrieve_file('ensembl', prompt=prompt)
                if ensembl_file is None:
                    print ('Unable to find or retrieve the ensembl file, ' 
                        'exiting')
                    return
            self.ensembl = pd.read_csv(ensembl_file, sep='\t')
        
        motif_df = self.all_edges.join(other=self.ensembl.set_index(
            'Transcript stable ID'), on='transcript')
        print ('Number of TF - transcript edges:', len(motif_df))
        if protein_coding_only:
            motif_df = motif_df[motif_df['Gene type'] == 
                'protein_coding'].copy()
        motif_df.drop(columns=['Gene type', 'name'], inplace=True)
        motif_df['TFName'] = motif_df['TFName'].apply(lambda x: 
            self.animal_to_human[x] if x in self.animal_to_human else x)
        motif_df.dropna(subset=['Gene stable ID'], inplace=True)
        motif_df.sort_values('score', ascending=False, inplace=True)
        motif_df.drop_duplicates(subset=['TFName', 'Gene stable ID'],
            inplace=True)
        print ('Number of TF - gene edges:', len(motif_df))
        if use_gene_names:
            motif_df['Gene name'] = motif_df.apply(lambda x: x['Gene name'] if 
                type(x['Gene name']) == str else x['Gene stable ID'], axis=1)
            name_id_matching = motif_df.groupby(
                ['Gene name', 'Gene stable ID'])['Gene name'].count()
            id_to_name = {i[1]: i[0] for i in name_id_matching.groupby(
                level=0).idxmax().values}
            motif_df['Gene name'] = motif_df['Gene stable ID'].apply(
                lambda x: id_to_name[x] if x in id_to_name else np.nan)
            motif_df.dropna(subset='Gene name', inplace=True)
            print ('Number of TF - gene edges after name conversion: ',
                len(motif_df))
        
        self.motif_frame = motif_df
        self.use_gene_names = use_gene_names
    

    def write_motif_prior(
        self,
        output_path: Union[str, bytes, os.PathLike] = 'motif_prior.tsv',
        use_gene_names: Optional[bool] = None,
        weighted: bool = False
    ) -> None:
        
        print ()
        print ('--- Running write_motif_prior() ---')

        if self.motif_frame is None:
            print ('No motif prior has been generated yet, please run '
                'aggregate_matches() first')
            return
        
        if use_gene_names is None:
            use_gene_names = self.use_gene_names
        if use_gene_names:
            column = 'Gene name'
        else:
            column = 'Gene stable ID'

        self.motif_frame.sort_values(by=['TFName', column], inplace=True)

        if weighted:
            self.motif_frame['weight'] = self.motif_frame['score'] / 100
            self.motif_frame[['TFName', column, 'weight']].to_csv(
                output_path, sep='\t', index=False, header=False)
        else:
            self.motif_frame['edge'] = 1
            self.motif_frame[['TFName', column, 'edge']].to_csv(
                output_path, sep='\t', index=False, header=False)


    def show_fingerprint(
        self
    ) -> None:
        
        for k,v in self.fingerprint.items():
            if v['provided']:
                print (f'{k}: provided by user')
            elif v['cached']:
                if v['version'] == '':
                    print (f'{k}: retrieved from cache')
                else:
                    print (f'{k}: {v["version"]}, retrieved from cache,',
                        'originally retrieved',
                        v['datetime'].strftime('%d/%m/%Y, %H:%M:%S'))                      
            else:
                print (f'{k}: {v["version"]}, retrieved',
                    v['datetime'].strftime('%d/%m/%Y, %H:%M:%S'))


    def clear_cache(
        self
    ) -> None:
        """
        Removes the temporary folder and everything in it.
        """
        
        if os.path.exists(self.temp_folder):
            rmtree(self.temp_folder)