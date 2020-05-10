#!/usr/bin/env python

def createIndex():
    filenames = ['./html/head.html',
                 './html/title.html',
                 './html/authors.html',
                 './html/school.html',
                 './html/image.html',
                 './html/abstract.html',
                 './html/paper.html',
                 './html/code.html',
                 './html/dataset.html',
                 # './html/demo.html',
                 './html/results.html',
                 './html/acknowledgements.html']
    with open('./index.html', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

def main():
    createIndex()

if __name__ == '__main__':
    main()
