import os
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    mkdir(outputdir)
    imgs = os.listdir(os.path.join(inputdir,'left'))
    for img in imgs:
        groups = ''
        # for idx in range(1, 101):
        #     groups = ''
        #     if idx == 1:
        #         groups += os.path.join(inputdir, vid, '00001' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00001' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00001' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00002' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00003' + ext) + '|'
        #         groups += os.path.join(targetdir, vid, '00001' + ext)
        #     elif idx == 2:
        #         groups += os.path.join(inputdir, vid, '00001' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00001' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00002' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00003' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00004' + ext) + '|'
        #         groups += os.path.join(targetdir, vid, '00002' + ext)
        #     elif idx == 99:
        #         groups += os.path.join(inputdir, vid, '00097' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00098' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00099' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00100' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00100' + ext) + '|'
        #         groups += os.path.join(targetdir, vid, '00099' + ext)
        #     elif idx == 100:
        #         groups += os.path.join(inputdir, vid, '00098' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00099' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00100' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00100' + ext) + '|'
        #         groups += os.path.join(inputdir, vid, '00100' + ext) + '|'
        #         groups += os.path.join(targetdir, vid, '00100' + ext)
        #     else:
        #         for i in range(idx-2, idx+3):
        #             groups += os.path.join(inputdir, vid, '{:05d}'.format(i) + ext) + '|'
        #         groups += os.path.join(targetdir, vid, '{:05d}'.format(idx) + ext)
        #     with open(os.path.join(outputdir, 'groups.txt'), 'a') as f:
        #         f.write(groups + '\n')
        groups += os.path.join(inputdir, 'left', img) + '|'
        groups += os.path.join(targetdir,'left', img) + '|'
        groups += os.path.join(inputdir, 'right',  img.replace('left','right')) + '|'
        groups += os.path.join(targetdir, 'right',  img.replace('left','right'))

        with open(os.path.join(outputdir, 'holopix0202_train.txt'), 'a') as f:
                f.write(groups + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/data/1760921465/Stereo_enhancement/holopix0202/train/low', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--target', type=str, default='/data/1760921465/Stereo_enhancement/holopix0202/train/gt', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='/code/STEN/data/', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.jpg', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()
