#import sys; sys.path.append('.')

import cv2, os, time
import numpy as np

from keras import backend as K
from pdb import set_trace as pause

from yad2k.keras_yolo import yolo_eval, yolo_head, loadYoloModel
from core.BoundBox import BoundBox
from core.Utils import convertDetectionsToMyImp, loadAndNormalizeImg

from core.DisplayUtils import put_title


def heat_map(activation_output, img, yolo_model):
    last_conv_layer = yolo_model.get_layer('leaky_re_lu_22')

    grads = K.gradients(activation_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([yolo_model.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([img])

    for j in range(1024):
        conv_layer_output_value[:, :, j] *= pooled_grads_value[j]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    heatmap = cv2.resize(heatmap, (img[0].shape[1], img[0].shape[0]))
    return heatmap


def heatmap_title(title, activation_output, img, yolo_model):
    print(activation_output)
    ht = heat_map(yolo_model.output[activation_output], img, yolo_model)
    return put_title(title, ht)

def normalize_grads(grds, groups):
    for group in groups:
        biggestValue = 0 
        for img_name  in group:

            if np.max(grds[img_name]) > biggestValue:
                biggestValue = np.max(grds[img_name])
           
        for img_name in group:
            grds[img_name] /= biggestValue
            grds[img_name] = cv2.resize(grds[img_name], (img[0].shape[1], img[0].shape[0]))
            grds[img_name] = np.uint8(255 * grds[img_name])
            grds[img_name] = cv2.applyColorMap(grds[img_name], cv2.COLORMAP_JET)
            grds[img_name] = grds[img_name].astype(float)/255

def alpha_grad(grds, img):
    for img_name  in grds:
        grds[img_name] = grds[img_name] * 0.75 + img * 0.5


def create_image_grid(grds, grid):
    final = np.array([])
    for group in grid:
        line = np.array([])
        for img_name in group:
            image =  grds[img_name]
            if line.size == 0:
                line = image
            else:
                line = np.concatenate((line,image), axis=1)
    
        if final.size == 0:
            final = line
        else:
            final = np.concatenate((final,line), axis=0)
    return final


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def getExperiments():
    experiments = {}
   

    key = "all"
    experiments[key] = {}
    experiments[key]["det"] = [["man","woman"]]
   
    experiments[key]["anc"]  = [["ach1_obj","ach2_obj","ach3_obj","ach4_obj", "ach1_man","ach2_man","ach3_man","ach4_man",
                          "ach1_wom","ach2_wom","ach3_wom","ach4_wom"]]

    # key = "indv"
    # experiments[key] = {}
    # experiments[key]["det"] = [["X"],["Y"],["X+Y"],["X+Y+W+H"],["W"],["H"],["W+H"],
    #                              ["objectness"],["man"],["woman"],["obj+man+woman"]]

    # experiments[key]["anc"]  = [["ach1_obj"],["ach2_obj"],["ach3_obj"],["ach4_obj"],
    #                             ["ach1_man"],["ach2_man"],["ach3_man"],["ach4_man"],
    #                             ["ach1_wom"],["ach2_wom"],["ach3_wom"],["ach4_wom"]]

    # key = "sep"
    # experiments[key] = {}
    # experiments[key]["det"] = [["X","Y"],["X+Y"],["X+Y+W+H"],["W","H"],["W+H"],
    #                              ["objectness"],["man","woman"],["obj+man+woman"]]

    # experiments[key]["anc"]  = [["ach1_obj","ach2_obj","ach3_obj","ach4_obj"],
    #                             ["ach1_man","ach2_man","ach3_man","ach4_man"],
    #                             ["ach1_wom","ach2_wom","ach3_wom","ach4_wom"]]

    return experiments

''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~ MAIN STARTS HERE, i think ~~~~~~~~~~~~~~~~~~~~~
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''


path_ = "/home/zeni/projects/in_progress/sibgraphi_2018/data/"

colors = {'man': (0.9019, 0.7647, 0.6235), 
          'woman': (0.6470, 0.3568, 1.0)}

images = ['000001.jpg', '000009.jpg', '000010.jpg', '000014.jpg', '000017.jpg', '000021.jpg', '000022.jpg', '000023.jpg', '000025.jpg', '000027.jpg', '000030.jpg', '000032.jpg', '000035.jpg', '000038.jpg', '000041.jpg', '000043.jpg', '000048.jpg', '000050.jpg', '000051.jpg', '000055.jpg', '000058.jpg', '000059.jpg', '000066.jpg', '000069.jpg', '000073.jpg', '000076.jpg', '000081.jpg', '000083.jpg', '000085.jpg', '000089.jpg', '000090.jpg', '000096.jpg', '000097.jpg', '000101.jpg', '000104.jpg', '000105.jpg', '000110.jpg', '000111.jpg', '000113.jpg', '000124.jpg', '000125.jpg', '000126.jpg', '000127.jpg', '000128.jpg', '000129.jpg', '000131.jpg', '000133.jpg', '000137.jpg', '000138.jpg', '000139.jpg', '000144.jpg', '000146.jpg', '000150.jpg', '000151.jpg', '000155.jpg', '000159.jpg', '000162.jpg', '000163.jpg', '000164.jpg', '000165.jpg', '000166.jpg', '000168.jpg', '000169.jpg', '000170.jpg', '000171.jpg', '000173.jpg', '000174.jpg', '000177.jpg', '000178.jpg', '000181.jpg', '000182.jpg', '000185.jpg', '000188.jpg', '000190.jpg', '000191.jpg', '000192.jpg', '000193.jpg', '000194.jpg', '000200.jpg', '000201.jpg', '000202.jpg', '000205.jpg', '000206.jpg', '000210.jpg', '000212.jpg', '000218.jpg', '000220.jpg', '000222.jpg', '000223.jpg', '000226.jpg', '000229.jpg', '000230.jpg', '000231.jpg', '000232.jpg', '000237.jpg', '000238.jpg', '000239.jpg', '000240.jpg', '000245.jpg', '000247.jpg', '000248.jpg', '000251.jpg', '000252.jpg', '000257.jpg', '000258.jpg', '000259.jpg', '000264.jpg', '000265.jpg', '000269.jpg', '000271.jpg', '000272.jpg', '000275.jpg', '000276.jpg', '000278.jpg', '000279.jpg', '000280.jpg', '000282.jpg', '000283.jpg', '000285.jpg', '000286.jpg', '000287.jpg', '000288.jpg', '000291.jpg', '000293.jpg', '000297.jpg', '000298.jpg', '000299.jpg', '000302.jpg', '000305.jpg', '000308.jpg', '000310.jpg', '000315.jpg', '000319.jpg', '000320.jpg', '000321.jpg', '000322.jpg', '000323.jpg', '000328.jpg', '000330.jpg', '000331.jpg', '000337.jpg', '000338.jpg', '000339.jpg', '000341.jpg', '000342.jpg', '000346.jpg', '000348.jpg', '000352.jpg', '000356.jpg', '000358.jpg', '000359.jpg', '000364.jpg', '000367.jpg', '000368.jpg', '000369.jpg', '000372.jpg', '000374.jpg', '000377.jpg', '000378.jpg', '000382.jpg', '000386.jpg', '000388.jpg', '000392.jpg', '000393.jpg', '000394.jpg', '000405.jpg', '000406.jpg', '000407.jpg', '000409.jpg', '000410.jpg', '000411.jpg', '000413.jpg', '000414.jpg', '000419.jpg', '000421.jpg', '000422.jpg', '000423.jpg', '000428.jpg', '000429.jpg', '000433.jpg', '000434.jpg', '000435.jpg', '000437.jpg', '000438.jpg', '000443.jpg', '000444.jpg', '000445.jpg', '000446.jpg', '000447.jpg', '000448.jpg', '000449.jpg', '000453.jpg', '000455.jpg', '000456.jpg', '000457.jpg', '000458.jpg', '000463.jpg', '000467.jpg', '000468.jpg', '000470.jpg', '000476.jpg', '000477.jpg', '000479.jpg', '000480.jpg', '000482.jpg', '000483.jpg', '000485.jpg', '000490.jpg', '000493.jpg', '000497.jpg', '000498.jpg', '000499.jpg', '000500.jpg', '000502.jpg', '000504.jpg', '000506.jpg', '000507.jpg', '000515.jpg', '000516.jpg', '000517.jpg', '000518.jpg', '000520.jpg', '000523.jpg', '000524.jpg', '000525.jpg', '000526.jpg', '000527.jpg', '000530.jpg', '000531.jpg', '000532.jpg', '000534.jpg', '000535.jpg', '000536.jpg', '000538.jpg', '000539.jpg', '000541.jpg', '000545.jpg', '000546.jpg', '000547.jpg', '000554.jpg', '000555.jpg', '000562.jpg', '000566.jpg', '000567.jpg', '000570.jpg', '000578.jpg', '000579.jpg', '000583.jpg', '000586.jpg', '000587.jpg', '000589.jpg', '000591.jpg', '000594.jpg', '000597.jpg', '000602.jpg', '000604.jpg', '000606.jpg', '000607.jpg', '000612.jpg', '000613.jpg', '000615.jpg', '000616.jpg', '000617.jpg', '000621.jpg', '000623.jpg', '000624.jpg', '000625.jpg', '000626.jpg', '000628.jpg', '000629.jpg', '000630.jpg', '000633.jpg', '000634.jpg', '000638.jpg', '000639.jpg', '000641.jpg', '000642.jpg', '000643.jpg', '000644.jpg', '000648.jpg', '000649.jpg', '000652.jpg', '000654.jpg', '000655.jpg', '000662.jpg', '000664.jpg', '000666.jpg', '000670.jpg', '000677.jpg', '000683.jpg', '000684.jpg', '000687.jpg', '000688.jpg', '000690.jpg', '000692.jpg', '000693.jpg', '000694.jpg', '000695.jpg', '000696.jpg', '000697.jpg', '000701.jpg', '000702.jpg', '000704.jpg', '000709.jpg', '000715.jpg', '000717.jpg', '000719.jpg', '000723.jpg', '000726.jpg', '000727.jpg', '000731.jpg', '000733.jpg', '000734.jpg', '000735.jpg', '000739.jpg', '000742.jpg', '000744.jpg', '000750.jpg', '000752.jpg', '000753.jpg', '000755.jpg', '000758.jpg', '000760.jpg', '000762.jpg', '000765.jpg', '000766.jpg', '000769.jpg', '000770.jpg', '000777.jpg', '000781.jpg', '000782.jpg', '000783.jpg', '000784.jpg', '000786.jpg', '000793.jpg', '000797.jpg', '000798.jpg', '000799.jpg', '000802.jpg', '000805.jpg', '000806.jpg', '000807.jpg', '000810.jpg', '000812.jpg', '000814.jpg', '000816.jpg', '000819.jpg', '000828.jpg', '000829.jpg', '000834.jpg', '000836.jpg', '000839.jpg', '000842.jpg', '000843.jpg', '000847.jpg', '000848.jpg', '000851.jpg', '000854.jpg', '000855.jpg', '000856.jpg', '000858.jpg', '000859.jpg', '000860.jpg', '000861.jpg', '000862.jpg', '000865.jpg', '000866.jpg', '000869.jpg', '000870.jpg', '000874.jpg', '000878.jpg', '000879.jpg', '000883.jpg', '000885.jpg', '000886.jpg', '000891.jpg', '000892.jpg', '000895.jpg', '000898.jpg', '000901.jpg', '000902.jpg', '000903.jpg', '000904.jpg', '000906.jpg', '000907.jpg', '000909.jpg', '000910.jpg', '000911.jpg', '000913.jpg', '000915.jpg', '000916.jpg', '000918.jpg', '000920.jpg', '000922.jpg', '000924.jpg', '000926.jpg', '000927.jpg', '000930.jpg', '000937.jpg', '000939.jpg', '000940.jpg', '000942.jpg', '000943.jpg', '000944.jpg', '000948.jpg', '000949.jpg', '000952.jpg', '000955.jpg', '000956.jpg', '000959.jpg', '000966.jpg', '000967.jpg', '000969.jpg', '000971.jpg', '000975.jpg', '000978.jpg', '000979.jpg', '000981.jpg', '000982.jpg', '000984.jpg', '000986.jpg', '000987.jpg', '000988.jpg', '000990.jpg', '000991.jpg', '000996.jpg', '000999.jpg', '001001.jpg', '001006.jpg', '001011.jpg', '001014.jpg', '001017.jpg', '001020.jpg', '001021.jpg', '001024.jpg', '001026.jpg', '001028.jpg', '001031.jpg', '001033.jpg', '001035.jpg', '001036.jpg', '001037.jpg', '001038.jpg', '001040.jpg', '001042.jpg', '001047.jpg', '001050.jpg', '001055.jpg', '001057.jpg', '001060.jpg', '001061.jpg', '001063.jpg', '001064.jpg', '001065.jpg', '001066.jpg', '001067.jpg', '001071.jpg', '001072.jpg', '001076.jpg', '001079.jpg', '001084.jpg', '001085.jpg', '001086.jpg', '001091.jpg', '001092.jpg', '001095.jpg', '001096.jpg', '001097.jpg', '001099.jpg', '001101.jpg', '001105.jpg', '001108.jpg', '001109.jpg', '001113.jpg', '001116.jpg', '001118.jpg', '001125.jpg', '001129.jpg', '001133.jpg', '001137.jpg', '001140.jpg', '001141.jpg', '001145.jpg', '001147.jpg', '001150.jpg', '001151.jpg', '001152.jpg', '001155.jpg', '001157.jpg', '001159.jpg', '001164.jpg', '001165.jpg', '001167.jpg', '001168.jpg', '001169.jpg', '001170.jpg', '001171.jpg', '001173.jpg', '001175.jpg', '001177.jpg', '001180.jpg', '001183.jpg', '001185.jpg', '001186.jpg', '001198.jpg', '001201.jpg', '001206.jpg', '001210.jpg', '001212.jpg', '001219.jpg', '001220.jpg', '001221.jpg', '001222.jpg', '001224.jpg', '001227.jpg', '001228.jpg', '001229.jpg', '001234.jpg', '001236.jpg', '001238.jpg', '001240.jpg', '001241.jpg', '001242.jpg', '001243.jpg', '001244.jpg', '001245.jpg', '001248.jpg', '001251.jpg', '001253.jpg', '001254.jpg', '001256.jpg', '001259.jpg', '001261.jpg', '001265.jpg', '001266.jpg', '001267.jpg', '001269.jpg', '001271.jpg', '001272.jpg', '001279.jpg', '001281.jpg', '001282.jpg', '001284.jpg', '001287.jpg', '001292.jpg', '001296.jpg', '001297.jpg', '001298.jpg', '001301.jpg', '001303.jpg', '001304.jpg', '001307.jpg', '001309.jpg', '001310.jpg', '001311.jpg', '001315.jpg', '001319.jpg', '001320.jpg', '001325.jpg', '001327.jpg', '001329.jpg', '001330.jpg', '001333.jpg', '001336.jpg', '001337.jpg', '001340.jpg', '001342.jpg', '001346.jpg', '001347.jpg', '001350.jpg', '001351.jpg', '001352.jpg', '001353.jpg', '001354.jpg', '001357.jpg', '001358.jpg', '001362.jpg', '001363.jpg', '001366.jpg', '001368.jpg', '001370.jpg', '001372.jpg', '001376.jpg', '001378.jpg', '001382.jpg', '001388.jpg', '001390.jpg', '001392.jpg', '001393.jpg', '001396.jpg', '001403.jpg', '001405.jpg', '001406.jpg', '001408.jpg', '001409.jpg', '001411.jpg', '001412.jpg', '001414.jpg', '001417.jpg', '001419.jpg', '001420.jpg', '001421.jpg', '001423.jpg', '001424.jpg', '001426.jpg', '001427.jpg', '001429.jpg', '001430.jpg', '001431.jpg', '001434.jpg', '001435.jpg', '001437.jpg', '001438.jpg', '001445.jpg', '001446.jpg', '001447.jpg', '001448.jpg', '001450.jpg', '001451.jpg', '001452.jpg', '001454.jpg', '001455.jpg', '001456.jpg', '001459.jpg', '001460.jpg', '001463.jpg', '001469.jpg', '001472.jpg', '001473.jpg', '001474.jpg', '001475.jpg', '001476.jpg', '001479.jpg', '001480.jpg', '001482.jpg', '001484.jpg', '001485.jpg', '001493.jpg', '001495.jpg', '001496.jpg', '001498.jpg', '001499.jpg', '001501.jpg', '001502.jpg', '001503.jpg', '001504.jpg', '001506.jpg', '001509.jpg', '001510.jpg', '001511.jpg', '001514.jpg', '001516.jpg', '001521.jpg', '001523.jpg', '001524.jpg', '001526.jpg', '001531.jpg', '001532.jpg', '001533.jpg', '001536.jpg', '001537.jpg', '001538.jpg', '001542.jpg', '001544.jpg', '001548.jpg', '001554.jpg', '001557.jpg', '001558.jpg', '001561.jpg', '001563.jpg', '001564.jpg', '001566.jpg', '001569.jpg', '001570.jpg', '001571.jpg', '001572.jpg', '001575.jpg', '001577.jpg', '001579.jpg', '001580.jpg', '001581.jpg', '001583.jpg', '001585.jpg', '001586.jpg', 'bad.jpg', 'japa.jpg', 'oldman.jpg', 'teste.jpg', 'vlcsnap-2018-06-15-19h48m26s397.png', 'vlcsnap-2018-06-15-19h48m35s666.png', 'woman.jpg']


enclosureFolder = time.strftime("%Y_%m_%d-%H_%M_%S") + "_heatmaps/"

local = path_ + "outputs/" + enclosureFolder 
if not os.path.exists(local):
    os.makedirs(local)


experiments = getExperiments()



#=====

for key in experiments:
    print (key)
    norm_grup_detections = experiments[key]["det"]

    norm_grup_anchors = experiments[key]["anc"]


    grid_detections = [["detection", "man","woman"]]
                     
                   
                       
    grid_anchors = [["original",      "ach1_obj","ach2_obj","ach3_obj","ach4_obj"],
                    ["original",      "ach1_man","ach2_man","ach3_man","ach4_man"],
                    ["all detections","ach1_wom","ach2_wom","ach3_wom","ach4_wom"]]


    for imgName in images:
        print (imgName)
        inputImage = path_ + 'test_images/' + imgName

        yolo_model, anchors, class_names = loadYoloModel(path_+'/model_tensorflow/gender_wild_vo50.h5', 
                                                     path_+'/model_tensorflow/yolo_anchors.txt', 
                                                     path_+'/model_tensorflow/gender_classes.txt')



        # yolo_model, anchors, class_names = loadYoloModel(path_+'/model_tensorflow/yoloGender_faceOnly_final.h5', 
        #                                          path_+'/model_tensorflow/yoloGender_faceOnly_final_anchors.txt',
        #                                          path_+'/model_tensorflow/gender_classes.txt')

        model_image_size = yolo_model.layers[0].input_shape[1:3]
        img = loadAndNormalizeImg(inputImage, model_image_size)

        sess = K.get_session() 

        yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
        input_image_shape = K.placeholder(shape=(2, ))

        boxes, scores, classes, boxes_scores = yolo_eval(
            yolo_outputs,
            input_image_shape,
            score_threshold=0.3,
            iou_threshold=0.5,
            grad_info=True)


        out_boxes, out_scores, out_classes, out_boxes_scores = sess.run(
            [boxes, scores, classes, boxes_scores],
            feed_dict={
                yolo_model.input: img,
                input_image_shape: [img.shape[1], img.shape[2]],
                K.learning_phase(): 0
            })
        print('Found {} boxes'.format(len(out_boxes)))

        detections = convertDetectionsToMyImp(out_boxes, out_scores, out_classes, class_names, img.shape)


        imgAllDetections=img[0].copy()

        for d in detections:
            d.drawInImage(imgAllDetections, scale = 4, color=colors[d.classId], lineWidth=3, text=d.classId, alpha = 0.7)
        #TODO: change to detections
        for i, d in enumerate(detections):
            
            print( out_boxes_scores.shape)
            #print(out_boxes_scores)
            print(i)
            print("score1", out_scores[i])
            l = np.where(out_boxes_scores == out_scores[i])
            print("score2", out_boxes_scores[l], out_boxes_scores.shape)

            print(l)
            print(out_boxes_scores[l[0][0],l[1][0],l[2][0],l[3][0],l[4][0]])

            bath_idx = l[0][0]
            x = l[1][0]
            y = l[2][0]
            anchor = l[3][0]
            best_class = l[4][0]
           
            loc = 7*anchor

            imgs =[]
            hts = []
            superimposed_imgs = []


            print(out_boxes_scores[l[0][0],l[1][0],l[2][0],l[3][0],0])
            print(out_boxes_scores[l[0][0],l[1][0],l[2][0],l[3][0],1])

            grds = {}
            grds["man"]        = heat_map(yolo_outputs[3][l[0][0],l[1][0],l[2][0],l[3][0],0], img, yolo_model)
            grds["woman"]      = heat_map(yolo_outputs[3][l[0][0],l[1][0],l[2][0],l[3][0],1], img, yolo_model)
           
      
            normalize_grads(grds, norm_grup_detections)
       
            detectionImg = img[0].copy()

            d.drawInImage(detectionImg, scale = 4, color=colors[d.classId], lineWidth=3, text=d.classId, alpha = 0.7)
           
            alpha_grad(grds, detectionImg)
    
            grds["detection"] = detectionImg
           
            print(classes.shape)
            detectionGrad = create_image_grid(grds, grid_detections)

            # detectionGrad = put_title("Normalization Groups: " + str(norm_grup_detections), detectionGrad, color = (0,0,0), background = (200,200,200), textScale=1.0,  widthScale = 2)
           
            if False:
                detectionGrad = image_resize(detectionGrad, height = 800, inter = cv2.INTER_AREA)
                anchors       = image_resize(anchors, height = 800, inter = cv2.INTER_AREA)

                #final = cv2.resize(final2,(1000,800))
                cv2.imshow("This detection Grads", detectionGrad)
                cv2.imshow("Anchors", anchors)
                key = cv2.waitKey(5000)
                if key == 27:
                    break
            else:

                local = path_ + "outputs/" + enclosureFolder +  key + "/"
                if not os.path.exists(local):
                    os.makedirs(local)

                cv2.imwrite(local + imgName.replace(".jpg", "") + "_detec_" +  str(i) + ".jpg", detectionGrad*255)
               
        sess.close()
        K.clear_session() 