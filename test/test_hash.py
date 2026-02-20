import os

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.channel_description as lps_channel_desc
import lps_synthesis.propagation.models as lps_models
import lps_synthesis.environment.acoustic_site as lps_as

def _main():
    output_dir = "./result/desc_test/"
    os.makedirs(output_dir, exist_ok=True)

    desc = lps_channel_desc.Description.get_random(seed=42)
    original_hash = hash(desc)
    print("Hash original:", original_hash)
    print(desc.__get_hash_base__())

    desc2 = lps_channel_desc.Description.get_random(seed=42)
    original_hash2 = hash(desc2)
    print("Hash original:", original_hash2)
    print(desc2.__get_hash_base__())


    file_path = os.path.join(output_dir,"test_description.json")
    desc.save(file_path)
    print(f"Description salvo em {file_path}")

    loaded_desc = lps_channel_desc.Description.load(file_path)
    loaded_hash = hash(loaded_desc)
    print("Hash após load:", loaded_hash)
    print(loaded_desc.__get_hash_base__())

    if original_hash == original_hash2:
        print("✅ Hashes iguais: instanciacao consistente")
    else:
        print("❌ Hashes diferentes: algo mudou na instanciacao")

    if original_hash == loaded_hash:
        print("✅ Hashes iguais: serialização consistente")
    else:
        print("❌ Hashes diferentes: algo mudou na serialização")

    config = lps_models.QueryConfig(description=desc, sensor_depth=lps_qty.Distance.m(30))
    config2 = lps_models.QueryConfig(description=desc2, sensor_depth=lps_qty.Distance.m(30))

    print("config:", hash(config))
    print("config2:", hash(config2))

    if config == config2:
        print("✅ QueryConfig iguais: independente da description")
    else:
        print("❌ QueryConfig diferentes: dependente da description")


    config = lps_as.AcousticSiteProspector.get_default_query(desc=desc,
                                                             sensor_depth=lps_qty.Distance.m(30))
    config2 = lps_as.AcousticSiteProspector.get_default_query(desc=desc2,
                                                              sensor_depth=lps_qty.Distance.m(30))
    config3 = lps_as.AcousticSiteProspector.get_default_query(desc=desc2,
                                                              sensor_depth=lps_qty.Distance.m(30.0))

    print("config:", hash(config))
    print("config2:", hash(config2))
    print("config3:", hash(config3))

    if config == config2:
        print("✅ AcousticSiteProspector.get_default_query iguais: independente da description")
    else:
        print("❌ AcousticSiteProspector.get_default_query diferentes: dependente da description")

    if config == config3:
        print("✅ AcousticSiteProspector.get_default_query iguais: independente da description")
    else:
        print("❌ AcousticSiteProspector.get_default_query diferentes: dependente da description")

if __name__ == "__main__":
    _main()
