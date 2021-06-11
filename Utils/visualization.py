from numpy.lib.arraysetops import isin
import visdom
import torch
import numpy

class VisdomVisualizer(object):
    def __init__(self, name, server="http://localhost", count=4):
        self.visualizer = visdom.Visdom(server=server, port=8097, env=name,\
            use_incoming_socket=False)
        self.name = name
        self.first_train_value = True
        self.first_test_value = True
        self.count = count
        self.plots = {}
        

    def show_parameters(self,params):
        text = ""
        for p in params:
            text += p + " : " + str(params[p]) + "<br>"
        
        self.visualizer.text(text=text, win="parameters")
        
    def append_loss(self, epoch, global_iteration, loss, loss_name="total", mode='train'):
        plot_name = loss_name + '_train_loss' if mode == 'train' else loss_name + '_test_loss'
        opts = (
            {
                'title': plot_name,
                #'legend': mode,
                'xlabel': 'iterations',
                'ylabel': loss_name
            })
        if mode == 'val':
            loss_value = float(loss)
        else:
            if isinstance(loss, torch.Tensor):
                loss_value = float(loss.detach().cpu().numpy())
            else:
                loss_value = loss
        
        if loss_name not in self.plots:
            self.plots[loss_name] = self.visualizer.line(X=numpy.array([global_iteration]), Y=numpy.array([loss_value]), opts=opts)
        else:
            self.visualizer.line(X=numpy.array([global_iteration]), Y=numpy.array([loss_value]), win=self.plots[loss_name], name=mode, update = 'append')
        
    def show_images(self, images, title):
        b, c, h, w = images.size()
        recon_images = images.detach().cpu()[:self.count, [2, 1, 0], :, :]\
            if c == 3 else\
            images.detach().cpu()[:self.count, :, :, :]
        opts = (
        {
            'title': title, 'width': self.count / 2 * 640,
            'height': self.count / 4 * 360
        })
        self.visualizer.images(recon_images, opts=opts,\
            win=self.name + title + "_window")

    def show_image(self, image, title):
        b, c, h, w = image.size()
        recon_image = image.detach().cpu()[0, [2, 1, 0], :, :]\
            if c == 3 else\
            image.detach().cpu()[:self.count, :, :, :]
        opts = (
        {
            'title': title, 'width': 320,
            'height': 180
        })
        self.visualizer.images(recon_image, opts=opts,\
            win=self.name + title + "_window")

    def show_activations(self, maps, title):
        c, h, w = maps.size()
        maps_cpu = maps.detach().cpu()[:, :, :]
        #maps_cpu = maps_cpu.squeeze(0)
        for i in range(c): #c
            opts = (
            {
                'title': title + str(i), 'colormap': 'Viridis'
            })
            heatmap = maps_cpu[i, :, :]
            heatmap_flipped = torch.flip(heatmap, [0])
            self.visualizer.heatmap(heatmap_flipped,\
                opts=opts, win=self.name + title + "_window_" + str(i))

    def show_seg_map(self, in_map, title, iter=0):
        maps = in_map.squeeze(0).detach()
        h, w = maps.size()  
        maps_cpu = maps.cpu()[:, :]
        opts = (
        {
            'title': title + "_" + str(iter), 'colormap': 'Viridis'
        })
        heatmap_flipped = torch.flip(maps_cpu, [0])
        self.visualizer.heatmap(heatmap_flipped,\
            opts=opts, win=self.name + title + "_window_")

    def show_kernels(self, maps, title):
        b, c, h, w = maps.size()
        maps_cpu = maps.detach().cpu()[:, :, :, :]
        maps_cpu = maps_cpu.squeeze(0)
        count, _, _ = maps_cpu.size()
        for i in range(count):
            opts = (
            {
                'title': title + str(i), 'colormap': 'Viridis'
            })
            heatmap = maps_cpu[i, :, :]
            self.visualizer.heatmap(heatmap,\
                opts=opts, win=self.name + title + "_window_" + str(i))

    def show_map(self, maps, title):
        b, c, h, w = maps.size()
        maps_cpu = maps.detach().cpu()[:self.count, :, :, :]
        for i in range(min(self.count, b)):
            opts = (
            {
                'title': title + str(i), 'colormap': 'Viridis'
            })
            heatmap = maps_cpu[i, :, :, :].squeeze(0)
            self.visualizer.heatmap(heatmap,\
                opts=opts, win=self.name + title + "_window_" + str(i))

    def show_point_clouds(self, coords, title):
        point_clouds = coords.detach().cpu()[:self.count, :, :, :]        
        opts = (
        {
            'title': title + '_points3D', 'webgl': True,
            #'legend'=['Predicted', 'Ground Truth'],
            'markersize': 0.5,
            #'markercolor': torch.tensor([[0,0,255], [255,0,0]]).int().numpy(),
            'xtickmin': -3, 'xtickmax': 3, 'xtickstep': 0.2,
            'ytickmin': -3, 'ytickmax': 3, 'ytickstep': 0.2,
            'ztickmin': -2, 'ztickmax': 5, 'ztickstep': 0.2
        })
        for i in range(self.count):
            p3d = point_clouds[i, :, :, :].permute(1, 2, 0).reshape(-1, 3)
            self.visualizer.scatter(X=p3d, opts=opts,\
                win=self.name + "_" + title + '_' + str(i+1))

    def show_point_cloud(self, coords, title):
        point_cloud = coords.detach().cpu()
        opts = (
        {
            'title': title + '_points3D', 'webgl': True,
            #'legend'=['Predicted', 'Ground Truth'],
            'markersize': 0.5,
            #'markercolor': torch.tensor([[0,0,255], [255,0,0]]).int().numpy(),
            'xtickmin': -1, 'xtickmax': 1, 'xtickstep': 0.2,
            'ytickmin': -0.3, 'ytickmax': 2, 'ytickstep': 0.2,
            'ztickmin': -1, 'ztickmax': 1, 'ztickstep': 0.2
        })
        self.visualizer.scatter(X=point_cloud, opts=opts,\
                win=self.name + "_" + title)

    def show_normals(self, normals_pred, title):
        normals = normals_pred.detach().cpu()
        normals = torch.abs(normals)
        #normals = normals.div(2) * 255

        #normals_step = normals.div_(torch.max(normals) - torch.min(normals))*255
        #normals_rescaled = normals_step.add(255)

        opts = (
            {
                'title': title
            })
        normals_flipped = torch.flip(normals, dims=[0])         #TODO: CHECK THIS
        self.visualizer.images(normals_flipped,\
                opts=opts, win=self.name + title + "_window_")