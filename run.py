def train_SRS_KD(model, config, optimizer,  train_dataloader, eval_dataloader):
    total_train_time = 0.

    loss_func = get_loss()

    for _epoch in trange(int(config['epochs']), desc="Epoch"):
        model.train()
        train_loss = 0
        correct = 0
        total = 0 

        logger.info("------------------------train-----------------------------")
        start = time.time()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            inputs, targets = batch
            inputs = inputs.to(args.device)
            targets = targets.reshape(-1).to(args.device)
            optimizer.zero_grad()
            
            loss = model(inputs)

            loss.backward()
        
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx==0 or (batch_idx+1) % max(10, batch_num//10)  == 0:
                logger.info("epoch: {}\t {}/{}".format(_epoch+1, batch_idx+1, batch_num))
                logger.info('Loss: %.3f | Acc(hit@1): %.3f%% (%d/%d)' % (
                    train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
             
            # break
                    

        end = time.time()
        total_train_time += end - start
        logger.info("Time for {}'th epoch: {} mins, time for {} epoches: {} hours".\
            format(_epoch+1, round((end - start) / 60, 2), _epoch+1, round(total_train_time / 3600, 2)))
        

        if _epoch >= args.eval_begin_epochs or _epoch % args.eval_per_epochs == 0:
            do_eval(model, eval_dataloader, args, _epoch+1)

        if args.shrink_lr:
            lr_scheduler.step()


def accuracy(output, target, topk=(5, 20)): # output: [batch_size, item_size] target: [batch_size]
    """Computes the accuracy over the k top predictions for the specified values of k"""
    global curr_preds_5
    global rec_preds_5
    global ndcg_preds_5
    global curr_preds_20
    global rec_preds_20
    global ndcg_preds_20

    for bi in range(output.shape[0]):
        pred_items_5 = utils.sample_top_k(output[bi], top_k=topk[0])  # top_k=5
        pred_items_20 = utils.sample_top_k(output[bi], top_k=topk[1])

        true_item=target[bi]
        predictmap_5={ch : i for i, ch in enumerate(pred_items_5)}
        pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

        rank_5 = predictmap_5.get(true_item)
        rank_20 = pred_items_20.get(true_item)
        if rank_5 == None:
            curr_preds_5.append(0.0)
            rec_preds_5.append(0.0)
            ndcg_preds_5.append(0.0)
        else:
            MRR_5 = 1.0/(rank_5+1)
            Rec_5 = 1.0#3
            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
            curr_preds_5.append(MRR_5)
            rec_preds_5.append(Rec_5)#4
            ndcg_preds_5.append(ndcg_5)  # 4
        if rank_20 == None:
            curr_preds_20.append(0.0)
            rec_preds_20.append(0.0)#2
            ndcg_preds_20.append(0.0)#2
        else:
            MRR_20 = 1.0/(rank_20+1)
            Rec_20 = 1.0#3
            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
            curr_preds_20.append(MRR_20)
            rec_preds_20.append(Rec_20)#4
            ndcg_preds_20.append(ndcg_20)  # 4