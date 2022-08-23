#collection of helpper functions


#The train data generator
class CustomDBDataGen(tf.keras.utils.Sequence):
    def __init__(self, conn, X_col, Y_col,
                 batch_size,
                 num_classes,
                 input_size=(28, 28, 1),
                 k,
                 ):

        #init db connection and init vars
        self.conn = conn
        self.test = test
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.k = k

        #randomly batch for first warm start
        if img_num / batch_size == 0:
            num_batches = int(img_num/batch_size)
        else:
            num_batches = 1 + int(img_num/batch_size)

        print('There are ',num_batches,' batches in the warm start')
        arr = []
        for n in range(num_batches):
            to_add = [n]*batch_size
            arr = arr + to_add
        
        arr = arr[:img_num] #Limit
        random.shuffle(arr) #shuffle

        #update the database
        curr = conn.cursor()
        for i, a in enumerate(arr):
            curr.execute('''UPDATE imgs SET batch_num = (?) WHERE id = (?)''',(int(a),int(i),))
        curr.commit()

    def on_epoch_end(self):
        #reset the batch numbers to -1 for all images
        curr = self.conn.cursor()
        curr.execute('''UPDATE imgs SET batch_num = -1''')
        curr.commit()
        


    def __getitem__(self, index):
        #select only the batch where batch = index
        curr = self.conn.cursor()
        curr.execute('''SELECT id,data, label_num FROM imgs WHERE batch_num = (?)''',(int(index),))
        ids = []
        imgs = []
        labels = []
        for id,img, label,used in curr:
            ids.append(id)
            imgs.append(img)
            labels.append(label)

        #convert to tensors
        ids = np.array(ids)
        ids = tf.cast(ids, 'int32')

        imgs = np.array(imgs)
        imgs = tf.cast(imgs,'float32')

        labels = np.array(labels)
        labels = tf.one_hot(labels,self.num_classes)

        return tuple([ids,imgs]), labels

    def __len__(self):
        #calculates the number of batches to use
        curr = self.conn.cursor()
        curr.execute('''SELECT MAX(batch_num) FROM imgs''')
        return curr.fetchone()[0] + 1

def setup_db(db_path,ds_name,train_percent,test_percent):

    def DB_add_img(conn, img):
        """
        Create a new img into the img table
        :param conn:
        :param img: (label_num,data,)
        """
        if conn is not None:
            sql = ''' INSERT INTO imgs(label_num,data)
                    VALUES(?,?) '''
            cur = conn.cursor()
            cur.execute(sql, img)

        else:
            print('ERROR connecting to db')

    sqlite3.register_adapter(np.ndarray, sf.array_to_bin)# Converts np.array to TEXT when inserting
    sqlite3.register_converter("array", sf.bin_to_array) # Converts TEXT to np.array when selecting

    # create a database connection
    conn_path = db_path + ds_name + str(random.randint(0,10000000))+'.db'
    conn = None
    try:
        conn = sqlite3.connect(db_file,detect_types=sqlite3.PARSE_DECLTYPES)
    except Error as e:
        print(e)
        return False

    #check table exists
    curr = conn.cursor()
    curr.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='imgs' ''')
    if curr.fetchone()[0]==1 :
        print('Table exists. Initilizing Data.')
        #initilise and reset data
        curr.execute(''' UPDATE imgs SET batch_num = (?) ''',('-1',))
        curr.commit()
    else:
        print('Table does not exist, building now...')
        sql_create_img_table = """ CREATE TABLE IF NOT EXISTS imgs (
                                        id INTEGER PRIMARY KEY,
                                        label_num INTEGER,
                                        data array,
                                        batch_num INTEGER,
                                        used INTEGER
                                    ); """

        # create table
        if conn is not None:
            try:
                c = conn.cursor()
                c.execute(create_table_sql)
            except Error as e:
                print(e)
        else:
            print("Error! cannot create the database connection.")
        
        #populate table
        #take the tfds dataset and produce a dataset and dataframe
        print('INIT: Using ',ds_name, ' data, downloading now...')

        train_split = 'train[:'+train_percent+'%]' 
        test_split = 'test[:'+test_percent+'%]'
        train_ds, ds_info = tfds.load(ds_name,with_info=True,shuffle_files=False,as_supervised=True,split=train_split)
        test_ds = tfds.load(ds_name,with_info=False,shuffle_files=False,as_supervised=True,split=test_split)

        #record ds metadata
        num_classes = ds_info.features['label'].num_classes
        class_names = ds_info.features['label'].names

        #Take the dataset and add info to db
        i = 0
        for image,label in train_ds:
            if i % 5000 == 0: print('Images Complete = ',i)
            data_to_add = (str(label.numpy()),image.numpy(),)
            DB_add_img(conn, data_to_add)
            i += 1
        conn.commit()

        #count amount of data avalible
        print('Total Stored Test and Train Data (from tfdata): ',ds_info[11])
    return

def setup_logs(group,config,disabled):
    #Setup logs and records
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    if disabled:
        os.environ['WANDB_DISABLED'] = 'true'
    wandb.login()
    wandb.init(project='k_diversity',entity='adamdowse',config=config,group=group)

def setup_model(model_name,learning_rate):
    tf.keras.backend.clear_session()
    model = sm.select_model(model_name,info.num_classes,info.img_shape)
    model.build(info.img_shape+1)
    model.summary()
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate),
    loss_func = keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc_metric = keras.metrics.CategoricalAccuracy()
    return model,optimizer,loss_func,train_loss,train_acc_metric,test_loss,test_acc_metric

def k_diversity(model,train_ds,k,batch_size):

    def calc_inner_diversity(current_indexes,i,grads):
        '''
        current_indexes = list of indexes in this batch [indexes]
        i = the item index that we are scoring
        grads = the list of aproximate gradients for this index [...]
        '''
        #calculate the euclidean distance between the last layer gradients
        new_dists = distance.cdist(grads[i],grads[current_indexes],metric='euclidean')
        return np.min(new_dists,axis=1)

    def calc_outer_diversity(current_indexes,batch_centers,i,grads,k):
        #calc the distance between the modified batch with the item included to the other batches
        #the score is 1/ the distance 
        new_batch_center = np.mean(np.append(grads[current_indexes],grads[i]),axis=1)
        print(new_batch_center)
        new_dists = distance.cdist(new_batch_center,np.delete(batch_centers,k,axis=0))
        new_dists = 1/new_dists
        return np.min(new_dists,axis=1)

    #collect the gradient information for all images
    #last layer gradients from coresets for data efficient training (logits - onehot_label)
    #TODO check this is doing the right thing
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output) #logits
    aprox_grads = []
    for img,label in train_ds:
        grads = intermediate_layer_model.predict(img)
        grads_to_add = []
        for g in grads:
            grads_to_add.append(g.numpy()-label.numpy())
        
        aprox_grads.append(np.flatten(grads_to_add)) # list of the grads [[grads],[grads],...]


    #run the submod evaluation
    avalible = np.ones(activations.shape[0]) # [1,1,1,1,1,...] denotes the items avalible 
    batch_indexes = np.zeros((k,batch_size)) -1 # [[batch0],[batch1],...,*k] init to -1

    #TODO work out how to store the batch diversitys (in and out) so we dont need to recompute each time
    #MAYBE a dictionary? TODO
    inner_div_store = np.zeros(k)
    outer_div_store = 0

    for b in range(batch_size): #each item goes into one batch at a time
        for batch_num in range(k):
            #calc inner batch distance
            current_batch_indexes = batch_indexes[batch_num]
            

            #calculate inner and outer diversity for all the possible new samples
            item_scores = []
            for i,grads in enumerate(aprox_grads): 
                if avalible[i] == 1:
                    #calc score for each sample
                    #TODO change the function inputs to match
                    inner_div_score = calc_inner_diversity(current_batch_indexes,inner_div_store,i,grads)
                    outer_div_score = calc_outer_diversity(current_batch_indexes)

                else:
                    #cant use this point but used to hold indexing position
                    item_scores.append(0) #TODO check this dosent break anything

            #combine the scores based on the parameters alpha and beta

            #pick greedly the best index and add it to the current batch


    #SHould end up with a list of lists of indexes of batches here

    #set the db values to the batch numbers of the images for each upcoming batch

