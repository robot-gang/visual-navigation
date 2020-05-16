#!/usr/bin/env python

def generate_index(filenames):
    with open('./index.html', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

def main():
    # paper_filenames = [
    #         './html/paper/head.html',
    #         './html/paper/title.html',
    #         './html/paper/authors.html',
    #         './html/paper/school.html',
    #         './html/paper/image.html',
    #         './html/paper/abstract.html',
    #         './html/paper/paper.html',
    #         './html/paper/code.html',
    #         './html/paper/dataset.html',
    #         './html/paper/demo.html',
    #         './html/paper/results.html',
    #         './html/paper/acknowledgements.html'
    #         ]
    proj_filenames = [
            './html/proj/head.html',
            './html/proj/title.html',
            './html/proj/authors.html',
            './html/proj/school.html',
            './html/proj/image.html',
            './html/proj/abstract.html',
            './html/proj/code.html',
            './html/proj/dataset.html',
            './html/proj/results.html',
            './html/proj/conclusion.html',
            './html/proj/bios.html',
            './html/proj/acknowledgements.html'
            ]
    # generate_index(paper_filenames)
    generate_index(proj_filenames)

if __name__ == '__main__':
    main()
