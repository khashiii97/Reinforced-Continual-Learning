flow_size = 100
pkt_size = 200

epochs = 50
task_epochs = 20
base_batch_sizes = [8,16,32,64,128]
task_batch_sizes = [4,16,64,128]
base_batch_size = 16
#base_batch_sizes = [32]
new_task_batch_size = 16
l1_lambda = 0.001
t_labels = ['benign', 'attack_bot', 'attack_DDOS',\
            'attack_portscan']
all_labels = ['vectorize_friday/benign', 'attack_bot', 'attack_DDOS',\
            'attack_portscan', 'Benign_Wednesday', 'DOS_SlowHttpTest',\
            'DOS_SlowLoris', 'DOS_Hulk', 'DOS_GoldenEye', 'FTPPatator',\
            'SSHPatator', 'Web_BruteForce', 'Web_XSS']
            
multi_label_class = [2,1,1,5,1,2,3]
            
            
###
divided1 = ['vectorize_friday/benign', 'attack_bot']
            
test1 = [ 'attack_DDOS','DOS_Hulk', 'DOS_GoldenEye', 'Benign_Wednesday']



            
test2 = ['SSHPatator', 'Web_BruteForce', 'Web_XSS','Benign_Wednesday']




####

divided2 = [ 'SSHPatator', 'Web_BruteForce', 'Web_XSS',\
             'Benign_Wednesday']
             
             
test3 = ['vectorize_friday/benign','attack_portscan','DOS_SlowHttpTest',\
            'DOS_SlowLoris', 'DOS_Hulk']



###
divided3 = ['attack_bot','attack_portscan',
             'Benign_Wednesday',]
             
             
test4 = ['attack_DDOS','vectorize_friday/benign','Benign_Wednesday']
