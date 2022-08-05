from ast import main
import os
#list all files in a directory to .txt file
def list_all_files(dir):
    files = []
    for r, d, f in os.walk(dir):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    return files
def main():
    files = list_all_files('./publicTest/pill/image')
    #export list of files to .txt file
    file = open('./test.txt', 'w')
    for i in files:
        file.write(i + '\n')
    file.close()


if __name__ == '__main__':
    main()