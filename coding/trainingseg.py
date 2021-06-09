from torch import torch
def training_seg(model,optimizer,train_dataloader,criterion):
    for epoch in range(20):
        bar = Bar('{}'.format(cfg.pytorch.exp_id[-60:]), max=20)
        running_Loss=0.0
        for i, data in enumerate(train_dataloader):
            images, labels=data
            optimizer.zero_grad()
            images=images.cuda('1')
            outputs=outputs.cuda('1')
            outputs=model(images)
            outputs=outputs.squeeze(2)
            #outputs=torch.argmax(outputs, dim=1)
            labels=labels.float()
            labels=torch.unsqueeze(labels, 1)
            print(type(labels))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_Loss+=loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    print('Finished Training')