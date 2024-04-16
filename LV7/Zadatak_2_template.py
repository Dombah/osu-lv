import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans


for i in range(5):
    # ucitaj sliku
    img = Image.imread(f"imgs\\test_{i+1}.jpg")

    # prikazi originalnu sliku
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    print(f'U slici ima {len(np.unique(img_array, axis=0))} jedinstvenih boja.') # 97924 boje


    k = 5
    km = KMeans(n_clusters = k, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(img_array)
    label = km.predict(img_array)
    
   

    # rezultatna cslika
    img_array_aprox = img_array.copy()


    for i in range(k):
        img_array_aprox[label == i] = km.cluster_centers_[i]

    img_aprox = np.reshape(img_array_aprox, (w, h ,d))

    # prikazi rezultantnu sliku
    plt.figure()
    plt.title("Rezultantna slika")
    plt.imshow(img_aprox)
    plt.tight_layout()
    plt.show()


    
    J_values = []
    for i in range(1,10):
        km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 5, random_state = 0)
        km.fit(img_array)
        J_values.append(km.inertia_)

    plt.figure()
    plt.plot(range(1,10),J_values, marker='o')
    plt.xlabel('Broj klastera')
    plt.ylabel('Vrijednost J')
    plt.title('Elbow metoda')
    plt.show()




