import sys, glob

def main():
    assert len(sys.argv) == 2
    dirname = sys.argv[1]
    obj_files = glob.glob(dirname + "/*.obj")
    print("Modifying:", obj_files)

    for obj_file in obj_files:
        readfile = open(obj_file, "r")
        lines = readfile.readlines()
        readfile.close()

        writefile = open(obj_file, "w")
        for line in lines:
            if line[0] != "f":
                writefile.write(line)
            else:
                split_line = line.strip().split(" ")
                if "//" in split_line[1]:
                    for i in range(1, 4):
                        ds_index = split_line[i].index("//")
                        split_line[i] = split_line[i][:ds_index + 1] \
                            + split_line[i][:ds_index] \
                            + split_line[i][ds_index + 1:]
                else:
                    for i in range(1, 4):
                        split_line[i] = split_line[i] + "/" + split_line[i] \
                            + "/" + split_line[i]
                modified_line = " ".join(split_line) + "\n"
                writefile.write(modified_line)
        writefile.close()

    print("Done modifying obj files!")

if __name__ == "__main__": main()
