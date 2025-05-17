def merge_bagz_files(input_files, output_file):
    with BagWriter(output_file) as writer:
        for input_file in input_files:
            reader = BagReader(input_file)
            for i in range(len(reader)):
                writer.write(reader[i])