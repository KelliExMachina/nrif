import os


def build_image_list(path):
    '''
    Search (path) and make a recursive listing of paths to each file.
    INPUTS: path = top of directory tree to begin recursion for images. ex) "../data"
    OUTPUTS: fname = A full path listing of all files in the directory tree.  astype.list()
    '''
    fname = []
    for root, d_names, f_names in os.walk(path):
        f_names = [f for f in f_names if not f[0] == '.']  # skip hidden files: .DSstore
        d_names = [d for d in d_names if not d[0] == '.']  # skip hidden folders: .git
        for f in f_names:
            fname.append(os.path.join(root, f))
    return fname
