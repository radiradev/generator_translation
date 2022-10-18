from src.root_dataloader import ROOTDataset

dataset = ROOTDataset(data_dir='/eos/home-c/cristova/DUNE/AlternateGenerators/', preload_data=True, generator_b='GiBUU')
# gibuu_data = dataset.load_generator('GiBUU')

print(dataset[0]['weights'])