

import os   ,cv2,numpy          
from PIL import Image                                #25.765                                  
from imagededup.methods import CNN
cnn_encoder = CNN()

rootdir = 'face_database'
count_img = 0
count_img_deleted = 0
def cls():
    os.system('cls' if os.name=='nt' else 'clear')
def remove_in_json(path, path_img):
    import json
    
    # Opening JSON file
    f = open(path_img,)
      
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
      
    # Iterating through the json
    # list
    for i in data:
        print(rootdir+'/'+path+'/'+i)
        os.remove(rootdir+'/'+path+'/'+i)
    cls()
    print ("Done")
    # Closing file
    f.close()
def remove_slow(rootdir):
    for it in os.scandir(rootdir):
        if it.is_dir():
            print(it.path)
            path_root_name = os.path.basename(it.path)
            path_img = path_root_name+'.json'
            if os.path.exists(it.path+'/'+path_img):
                os.remove(it.path+'/'+path_img)
            duplicates = cnn_encoder.find_duplicates_to_remove(image_dir=it.path, min_similarity_threshold=0.85,outfile=path_img)
            remove_in_json(path_root_name,path_img)
            
def check_identical_cv2(path_img_1, path_img_2):
    original = cv2.imread(path_img_1)
    duplicate = cv2.imread(path_img_2)
    if original.shape == duplicate.shape:
        difference = cv2.subtract(original, duplicate)  
         
        result = not numpy.any(difference)
        
        b, g, r = cv2.split(difference)
        if result is True and cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            cv2.waitKey(1)
            cv2.imwrite("ed.jpg", difference)
            return True
        return False    
    else:
        return False

def check_identical_PIL(path_img_1, path_img_2):
    img1 = Image.open(path_img_1)
    img2 = Image.open(path_img_2)
    
    if list(img1.getdata()) == list(img2.getdata()):

        return True
    else:
    
        return False
                
def remove_fast(rootdir):
    global count_img_deleted, count_img
    subfolders= [x[0] for x in os.walk(rootdir)]
    print(subfolders)
    
    for r1, d1, f1 in os.walk(rootdir):
        for file_v1 in f1:
    
            if '.jpg' in file_v1:
                # print("Kieru " + os.path.join(r, file) + str(i))
                path_img_1 = os.path.join(r1, file_v1)
                count_img += 1
                for r2, d2, f2 in os.walk(rootdir):
                    for file_v2 in f2:
               
                        if '.jpg' in file_v2 and os.path.join(r1, file_v1) != os.path.join(r2, file_v2):
                            
                            path_img_2 = os.path.join(r2, file_v2)
                            if check_identical_PIL(path_img_1,path_img_2) or check_identical_cv2(path_img_1,path_img_2):
                                count_img_deleted += 1
                                print ("Identical")
                                if os.path.exists(path_img_2):                      
                                    os.remove(path_img_2)
                                    cls()
                            else:
                                cls()
                                print ("Processing...")
                                print ("Total img: " + str(count_img))
                                print ("Total img deleted: " + str(count_img_deleted))
                                
    print ("Done!")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #remove_fast(rootdir)
    remove_slow(rootdir)
    #remove_in_json("surprised","my_duplicates.json")
