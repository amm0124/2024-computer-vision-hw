import hw_utils as utils
import matplotlib.pyplot as plt


def main():
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/basmati', ratio_thres=0.6)
    #im_book = utils.Match('./data/scene', './data/book', ratio_thres=0.6)
    #im_box = utils.Match('./data/scene', './data/box', ratio_thres=0.6)
    plt.title('Match')
    plt.imshow(im)
    #im_book.save('./result/part1-1-book.png','PNG')
    #im_box.save('./result/part1-1-box.png','PNG')
    
    # Test run matching with ransac //// part 1-2
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/library', './data/library2',
        ratio_thres=0.6, orient_agreement=30, scale_agreement=0.5)
        #ratio_thres=0.45, orient_agreement=20, scale_agreement=0.5)
        #ratio_thres=0.45, orient_agreement=30, scale_agreement=0.4) 
    
    plt.title('MatchRANSAC')
    plt.imshow(im)
    #im.save('./result/part1-2-library.png', 'PNG') 
    
if __name__ == '__main__':
    main()
