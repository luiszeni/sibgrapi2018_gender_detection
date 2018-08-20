#include "region_attr_layer.h"
#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_region_attr_layer(int batch, int w, int h, int n, int classes, int coords, int attributes, int landmarks, int max_boxes)
{

    layer l = {0};
    l.type = REGION_ATTR;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + coords + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.coords = coords;
    l.attributes = attributes;
    l.landmarks = landmarks;
    l.max_boxes = max_boxes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n*2, sizeof(float));
    l.bias_updates = calloc(n*2, sizeof(float));
    
                                            //adding tot the final volume the attributes
    l.outputs = h*w*n*(classes + coords + 1) + h*w*attributes;
    l.inputs = l.outputs;

    //This MOTHAFUCHER!
    l.truths = max_boxes*(l.coords*l.classes + l.attributes + l.landmarks*2);
    
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_attr_layer;
    l.backward = backward_region_attr_layer;
#ifdef GPU
    l.forward_gpu = forward_region_attr_layer_gpu;
    l.backward_gpu = backward_region_attr_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection with attributes ;)\n");
    srand(0);

    return l;
}

void resize_region_attr_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_region_attr_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;

    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    //printf("Xs0:%f, Xs1:%f, box = x%f, y%f\n\n",i + x[index + 0*stride],j + x[index + 1*stride], b.x, b.y );

    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    //printf("Xs0:%f, Xs1:%f, box = w%f, h%f\n\n",exp(x[index + 2*stride]),exp(x[index + 3*stride]), b.w, b.h );
    return b;
}
 


 //   delta_region_box(truth,    l.output, l   .biases,    n,     box_index, i,      j,     l.w, l.h,      l.delta,            .    01,      l.w*l.h);
float delta_region_attr_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{

    box pred = get_region_attr_box(x, biases, n, index, i, j, w, h, stride);
    
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

void delta_region_attr_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[index + i*stride] = scale*(truth[i] - x[index + i*stride]);
    }
}


void delta_region_attr_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class >= 0){

            pred *= output[index + stride*class];
            int g = hier->group[class];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);

            class = hier->parent[class];
        }
        *avg_cat += pred;
    } else {
        if (delta[index] && tag){
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);
            return;
        }

        //For each class.
        for(n = 0; n < classes; ++n){

            //Negativates if not the class stride,  if not remove error ?
            delta[index + stride*n] = scale * (((n == class)?1 : 0) - output[index + stride*n]);
            
            // receives the value of the classification ?
            if(n == class) 
                *avg_cat += output[index + stride*n];
        }
    }
}


void forward_region_attr_layer(const layer l, network net)
{
    //printf("forward_region_layer\n");
    int i,j,b,t,c,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;


    //printf("==================~forward_region_attr_layer~=================================================\n");
    for (b = 0; b < l.batch; ++b) {

        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    
                    //Index of the first value in the vector of this n anchorbox
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    
                  
                    //returns the predicted box in the i,j position of anchor n
                    box pred = get_region_attr_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                    
                    //searches for the best IOU between the predicted box and ground truth
                    float best_iou = 0;
                    int moreGT = 1;
                    for(t = 0; t < 30; ++t){//I am asking myself why he choses 30 as max GT boxes...
                        if(!moreGT) break;
                        for(c = 0; c < l.classes; ++c){

                            box truth = float_to_box(net.truth + c*l.coords + t*(l.coords*l.classes + l.attributes + l.landmarks*2) + b*l.truths, 1);

                            // if there is no GT or no more GT boxes, ends the loop
                            if(!truth.x){
                                    if(c == 0) moreGT = 0;
                                    break;
                            } 
                            float iou = box_iou(pred, truth);
                            if (iou > best_iou) {
                                best_iou = iou;
                            }
                        }
                    }


                   
                    //Gets the 4th entri in the region box, whitch I think os the objectness stride 
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords);

                    //adds the value of this bad boy in the avg_anyobj
                    avg_anyobj += l.output[obj_index]; 


                    /*QUESTION: what is the utility of the noobject_scale?
                        I think that it goes negative if not have an object there.
                        noobject_scale = how mutch it will be negativated it is a kind of penality
                    */
                    
                    //Negativates noobject in the stride of objectness, 
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
           
                    //QESTION, what is the utility of the bacground in this scenario ?
                    if(l.background){
                        l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
                    }
                   

                    //In config file a threshold is setted in l.thresh, if this pred box best_io is greater than l.thresh
                    //If there is an object there, obj_idx in delta will be setted to 0.
                    if (best_iou > l.thresh) {
                        l.delta[obj_index] = 0;
                    }



                    //what the fuck is this?
                    //Aparentlly there is a limit in how muck images the network will do the following task
                    if(*(net.seen) < 12800){
                        

                        box truth = {0};

                        //Create the "true"  anchor box for this N Box vollum
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n]/l.w;
                        truth.h = l.biases[2*n+1]/l.h;
                        
                        //Populates the delta with de box truth + contribution of the values in X.
                        //sounds that the model is trying to converge to correct box in this N
                        delta_region_attr_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
                    }
                }
            }
        }

        //---------------
        //Until here it updated the Delta with the truth boxes
        //---------------



        //02/01/2018 - It have to deal with two bounding boxes
        //for each GT BOX
        int moreGT = 1;
        for(t = 0; t < 30; ++t){
            if(!moreGT) break;
            for(c = 0; c < l.classes; ++c){

                box truth = float_to_box(net.truth + c*l.coords + t*(l.coords*l.classes + l.attributes + l.landmarks*2) + b*l.truths, 1);

                if(!truth.x){
                    if(c == 0) moreGT = 0;
                    break;
                } 

                float best_iou = 0;
                int best_n = 0;
                
                //Get position of this GTBox in the vollum
                i = (truth.x * l.w); 
                j = (truth.y * l.h);
                


                //not sure of the utility of this
                box truth_shift = truth;
                truth_shift.x = 0;
                truth_shift.y = 0;
               

                
                //Find the anchor box with best IOU with the GT
                for(n = 0; n < l.n; ++n){

                    //Get the index from where this box have to be in each N 
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_region_attr_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                    
                    // true in cfg file
                    if(l.bias_match){
                        //update w/h of the predict using he biases.
                        pred.w = l.biases[2*n]/l.w;
                        pred.h = l.biases[2*n+1]/l.h;
                    }

                    //Basicaly it centralizes the X and Y from  both predicted and truth to calculate the iou
                    pred.x = 0;
                    pred.y = 0;

                    float iou = box_iou(pred, truth_shift);
                    if (iou > best_iou){
                        best_iou = iou;
                        best_n = n;
                    }
                }

                //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);

                //Get index of the best Anchor Box
                int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
                

                //update the delta, and get the IOU with the GT BOX
                float iou = delta_region_attr_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
               

                //If coords > 4 ?
                //maybe this will be useful for the aplication that I am Thinking
                //TODO: Study thios in future 
                if(l.coords > 4){
                    int mask_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 4);
                    delta_region_attr_mask(net.truth + t*(l.coords + 1) + b*l.truths + 5, l.output, l.coords - 4, mask_index, l.delta, l.w*l.h, l.mask_scale);
                }



                //Counts Recall =)
                if(iou > .5) 
                    recall += 1;


                //Avg IOU -> usefull to see how much the method is hiting correctlly
                avg_iou += iou;



                //l.delta[best_index + 4] = iou - l.output[best_index + 4];

                //gets the objctness index of the GT box
                int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
                
                //Objectness mean
                avg_obj += l.output[obj_index];
                
                //update to a larger value, it will influenciate the confidence of an object here
                l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
                

                //Rescore using the IOU? not sure the consequence of this
                if (l.rescore) {
                    l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
                }


                //BG is off
                if(l.background){
                    l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
                }


                //Get correct class of this BB
                int class = c;
                

                //MAP is off
                if (l.map)
                    class = l.map[class];
                

                //index of first class ?
                int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);
               
                //update delta  to get the probability of each class.
                delta_region_attr_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);

                ++count;
                ++class_count;
            }
        }
    }


    //03/01/18 = aparentemente deve conseguir trienar co BBxes agora mesmo com o volume esperando + 40 atributos...




    //printf("\n");
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);

    printf("|\tRegion Avg IOU: %f, ", avg_iou/count );
    printf("Class: %f, ", avg_cat/class_count );
    printf("Obj: %f, ", avg_obj/count );
    printf("No Obj: %f, ", avg_anyobj/(l.w*l.h*l.n*l.batch) );
    printf("Avg Recall: %f, ",  recall/count);
    printf("count: %d, ", count);
    printf("cost: %f\n", *(l.cost) );
}

void backward_region_attr_layer(const layer l, network net)
{
    //printf("backward_region_layer\n");
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
}

void correct_region_attr_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = boxes[i];
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}

void get_region_attr_boxes(layer l, int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, float **masks, int only_objectness, int *map, float tree_thresh, int relative)
{
    int i,j,n,z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w/2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for(z = 0; z < l.classes + l.coords + 1; ++z){
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if(z == 0){
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for(i = 0; i < l.outputs; ++i){
            l.output[i] = (l.output[i] + flip[i])/2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j){
                probs[index][j] = 0;
            }
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            boxes[index] = get_region_attr_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            if(masks){
                for(j = 0; j < l.coords - 4; ++j){
                    masks[index][j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
                if(map){
                    for(j = 0; j < 200; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    int j =  hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    probs[index][j] = (scale > thresh) ? scale : 0;
                    probs[index][l.classes] = scale;
                }
            } else {
                float max = 0;
                for(j = 0; j < l.classes; ++j){
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                    float prob = scale*predictions[class_index];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                    if(prob > max) max = prob;
                    // TODO REMOVE
                    // if (j == 56 ) probs[index][j] = 0; 
                    /*
                       if (j != 0) probs[index][j] = 0; 
                       int blacklist[] = {121, 497, 482, 504, 122, 518,481, 418, 542, 491, 914, 478, 120, 510,500};
                       int bb;
                       for (bb = 0; bb < sizeof(blacklist)/sizeof(int); ++bb){
                       if(index == blacklist[bb]) probs[index][j] = 0;
                       }
                     */
                }
                probs[index][l.classes] = max;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
    correct_region_boxes(boxes, l.w*l.h*l.n, w, h, netw, neth, relative);
}

#ifdef GPU

void forward_region_attr_layer_gpu(const layer l, network net)
{
    //Copy   size              from               to
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    

    int b, n;
    
    //printf("Activations in Anchor boxes Region\n");
    //for each anchor box in each batch
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            
            //Get the adress to this anchor of this batch in the block
            int index = entry_index(l, b, n*l.w*l.h, 0);
            //printf("\tIndex of anchors %d in batch %d is %d and block size is %d \n",n, b, index, 2*l.w*l.h  );
            //Activate LOGISTIC for the X an Y coords. (Why only X and Y  and  why LOGISTIC)
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            
            //not the case
            if(l.coords > 4){
                index = entry_index(l, b, n*l.w*l.h, 4);
                activate_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC);
            }
            
            //Get index for the Objectness
            index = entry_index(l, b, n*l.w*l.h, l.coords);
                       
            //actrivate logiistic in the  obectness stride of this box
            if(!l.background) 
                activate_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC);
            
            
            //get the initial adress of the classes 
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            
            // not the case, as we are softmax
            if(!l.softmax && !l.softmax_tree) {
                activate_array_gpu(l.output_gpu + index, l.classes*l.w*l.h, LOGISTIC);
            }
        }


    }

   // printf("Activations in Attributes  Region\n");
    //TODO re-join these fors,  now this is only to see indexes separated
    for (b = 0; b < l.batch; ++b){
        //Activates the attributes part

        //Get index too the attributes part
        int index = entry_index(l, b, l.n*l.w*l.h, 0);
      //  printf("\tIndex of attributes in batch %d is %d and block size is %d \n",b, index, l.attributes*l.w*l.h  );
        //Logistic into this mothafucjers
        activate_array_gpu(l.output_gpu + index, l.attributes*l.w*l.h, LOGISTIC);


    }

    //not the case
    if (l.softmax_tree){
        int index = entry_index(l, 0, 0, l.coords + 1);           
        softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, 1, l.output_gpu + index, *l.softmax_tree);
    // we are SOFTMAX MOTHAFUCKER
    } else if (l.softmax) {


        // necessary to do for each bath the softmax because I added the attribute layers in each bach.
        //TODO: Analyze the function softmax_gpu and the group offset to remove this for.
    //    printf("Softmax in classes of Anchor boxes\n");
        for (b = 0; b < l.batch; ++b){
            // get the initial adress of classes
            int index = entry_index(l, b, 0, l.coords + !l.background);
           
            //relly softmaz (aqui)

                            
            softmax_gpu(
                //input volum to softmax         | OK
                net.input_gpu + index,     
                //number of classes + 0          | OK  ,
                l.classes + l.background,     
                //int batch,  number of anch. boxes in batch | OK
                l.n,            
                // int batch_offset,             |
                l.w*l.h*(l.coords + l.classes + 1),      
                //in groups,                     | OK
                l.w*l.h,  
                //group_offset,                  | OK
                1,
                //stride                         | OK
                l.w*l.h, 
                //temp,                          | OK
                1, 
                //Where save the softmax result  | OK
                l.output_gpu + index);


           //  printf("\tIndex %d -  batch %d batch offset %d \n", index, l.n, l.w*l.h*(l.coords + l.classes + 1));
        }

    }


    //if not training it stop here, copyng the output to ram memory...
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_region_attr_layer(l, net);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    if(!net.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_region_attr_layer_gpu(const layer l, network net)
{
    //printf("backward_region_layer_gpu\n" );
    

    int b, n;

    //For each Anchor block in each bach
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){

            //Get the index of this anchor in the Block
            int index = entry_index(l, b, n*l.w*l.h, 0);


            // Aparently the gradient only optimizes the X and Y in the block
            //GRADIENT ------| Adress of this an.box | Size of block| function?  |    Adress of delta  
            gradient_array_gpu(l.output_gpu + index,   2*l.w*l.h,    LOGISTIC,     l.delta_gpu + index);
            


            //  Only do Gradient if there are more than 4 coords
            if(l.coords > 4){
                index = entry_index(l, b, n*l.w*l.h, 4);
                gradient_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            }
            

            //get the indes of the objectness
            index = entry_index(l, b, n*l.w*l.h, l.coords);

            //Optimizes the OBjectness
            if(!l.background) 
                gradient_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC, l.delta_gpu + index);
        }
    }

    //I think it copyes the delta from other place in the GPU Memory.
    //          int N,         float ALPHA, float * X,    int INCX,   float * Y,     int INCY);
    axpy_gpu(l.batch*l.inputs, 1,            l.delta_gpu,  1,         net.delta_gpu, 1);
}
#endif
