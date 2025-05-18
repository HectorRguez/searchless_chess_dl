# from searchless_chess.src.bagz import BagReader, BagWriter
# import re

# def merge_specific_shards(file_pattern, start_shard, end_shard, output_file):
#     """
#     Merge a specific range of sharded bagz files.
    
#     Args:
#         file_pattern: Pattern for the sharded files with {} for shard number
#         start_shard: Starting shard index (inclusive)
#         end_shard: Ending shard index (exclusive)
#         output_file: Path to the output merged file
#     """
#     input_files = []
    
#     # Generate the specific shard filenames
#     for shard_idx in range(start_shard, end_shard):
#         shard_filename = file_pattern.format(f"{shard_idx:05d}")
#         input_files.append(shard_filename)
    
#     print(f"Merging {len(input_files)} files from shard {start_shard} to {end_shard-1}")
    
#     with BagWriter(output_file) as writer:
#         for input_file in input_files:
#             print(f"Processing {input_file}")
#             try:
#                 reader = BagReader(input_file)
#                 for i in range(len(reader)):
#                     writer.write(reader[i])
#                 print(f"Completed {input_file}")
#             except Exception as e:
#                 print(f"Error processing {input_file}: {e}")
#                 # Continue with the next file if one fails

# # Usage example for the first 256 shards
# file_pattern = "/data/hector/searchless_chess/train/action_value-{}-of-02148_data.bag"
# start_shard = 0
# end_shard = 256  # Will process shards 0-255
# output_file = "/data/hector/searchless_chess/train/action_value_data.bag"

# merge_specific_shards(file_pattern, start_shard, end_shard, output_file)

from searchless_chess.src.bagz import BagReader, BagWriter
import re

def merge_additional_shards(existing_file, file_pattern, start_shard, end_shard, output_file):
    """
    Merge additional shards into an existing merged file.
    
    Args:
        existing_file: Path to the existing merged file
        file_pattern: Pattern for the sharded files with {} for shard number
        start_shard: Starting shard index (inclusive)
        end_shard: Ending shard index (exclusive)
        output_file: Path to the output merged file
    """
    # First, create a temporary file with just the new shards
    temp_file = output_file + ".temp"
    input_files = []
    
    # Generate the specific shard filenames for the new range
    for shard_idx in range(start_shard, end_shard):
        shard_filename = file_pattern.format(f"{shard_idx:05d}")
        input_files.append(shard_filename)
    
    print(f"Merging {len(input_files)} additional files from shard {start_shard} to {end_shard-1}")
    
    # Merge the new shards into a temporary file
    with BagWriter(temp_file) as writer:
        for input_file in input_files:
            print(f"Processing {input_file}")
            try:
                reader = BagReader(input_file)
                for i in range(len(reader)):
                    writer.write(reader[i])
                print(f"Completed {input_file}")
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
    
    print(f"Finished creating temporary file with new shards")
    
    # Now merge the existing file and the temporary file
    print(f"Merging existing file and new shards into final output")
    with BagWriter(output_file) as writer:
        # First copy all records from the existing file
        print(f"Copying data from existing file: {existing_file}")
        existing_reader = BagReader(existing_file)
        total_existing = len(existing_reader)
        print(f"Found {total_existing} records in existing file")
        
        for i in range(total_existing):
            if i % 100000 == 0 and i > 0:
                print(f"Copied {i}/{total_existing} records from existing file")
            writer.write(existing_reader[i])
        
        # Then copy all records from the temporary file
        print(f"Copying data from temporary file: {temp_file}")
        temp_reader = BagReader(temp_file)
        total_temp = len(temp_reader)
        print(f"Found {total_temp} records in temporary file")
        
        for i in range(total_temp):
            if i % 100000 == 0 and i > 0:
                print(f"Copied {i}/{total_temp} records from temporary file")
            writer.write(temp_reader[i])
    
    print(f"Successfully merged existing data with new shards to {output_file}")
    print(f"Total records: {total_existing + total_temp}")
    
    # Clean up the temporary file
    import os
    os.remove(temp_file)
    print(f"Removed temporary file: {temp_file}")

# Usage example for adding shards 256-300 to the existing merged file
existing_file = "/data/hector/searchless_chess/train/action_value_data.bag"
file_pattern = "/data/hector/searchless_chess/train/action_value-{}-of-02148_data.bag"
start_shard = 301
end_shard = 401  # Will process shards 256-300
output_file = "/data/hector/searchless_chess/train/action_value_data_extended.bag"

merge_additional_shards(existing_file, file_pattern, start_shard, end_shard, output_file)