def resize_for_unet(X,image_size,deep_unet): 
    """
    prend 2 tenseurs d'ordre 3 avec le même shape: image_size
    et découpe les bords pour que la taille (hauteur & largeur)
    de chaques images soit un muliple de 16
    """
    
    resize_x=image_size[0]%(2**deep_unet)
    resize_y=image_size[1]%(2**deep_unet)

    x_droite=resize_x//2
    x_gauche=resize_x-x_droite
    y_haut=resize_y//2
    y_bas=resize_y-y_haut

    if (resize_x,resize_y)==(0,0):
        return X
    
    X1=[]
    for i in range(x_droite,image_size[0]-x_gauche):
        M=[]
        for j in range(y_haut,image_size[1]-y_bas):
            M.append(X[i,j,:])
        X1.append(M)
    return X1
