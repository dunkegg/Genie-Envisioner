import cv2
import time
import random
import pygame
import numpy as np
import networkx as nx
from termcolor import cprint


class GraphMapGenerator:
    """
    GraphMapGenerator
      mode
        1 - Initialization : start with blank
        2 - Read from array (N * N): 1 : free space, 0 : obstacle # TODO: DIFFERENT SEMANTICS
        3 - Read from image (N * N): white : free space, black : obstacle
        
    """
    def __init__(self, img=None, array=None, 
                 map_size=30, grid_size=80, 
                 obs_color=(0, 0, 0), free_color=(255, 255, 255), 
                 map_margin=1,
                 if_vis=True):
        
        self.obs_color = obs_color
        self.free_color = free_color
        
        self.grids = np.array([[0 for _ in range(map_size)] for _ in range(map_size)], dtype=int)
        self.map_size = map_size
        self.map_margin = map_margin
        self.grid_size = grid_size
                
        self.mode = 1
        if img is not None:
            cprint("Mode 3: Read from Image ! ", color='green', attrs=['bold'])
            self.load_from_img(img)
            self.mode = 2
        elif array is not None:
            cprint("Mode 2: Read from Array ! ", color='green', attrs=['bold'])
            self.load_from_array(array)
            self.mode = 3
        else:
            cprint("Mode 1: Start from Blank ! ", color='green', attrs=['bold'])

        self.window_size = grid_size * self.map_size
        
        if if_vis:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Draw Grid Map")

    def load_from_img(self, img):
        h, w = img.shape[0], img.shape[1]
        self.map_size = int(h / self.grid_size)
        grid_h, grid_w = h / self.map_size, w / self.map_size
        map = np.zeros((self.map_size, self.map_size), dtype=int)
        for i in range(self.map_size):
            for j in range(self.map_size):
                t = img[round(h * i / self.map_size + 0.5 * grid_h), 
                        round(w * j / self.map_size + 0.5 * grid_w), :]
                if t[0] >= 250 and t[1] >= 250 and t[2] >= 250: # 255 <=> free space
                    map[i,j] = 1
        self.grids = map
        
    def load_from_array(self, array):
        self.grids = array
        self.map_size = self.grids.shape[0]
        
    def load_from_networkx(self, graph):
        self.grids = np.array([[0 for _ in range(self.map_size)] for _ in range(self.map_size)], dtype=int)
        for idx, nx_node in enumerate(graph.nodes):            
            pos =  np.array(nx_node) 
            self.grids[pos[0]][[pos[1]]] = 1
        
    def to_graph(self, if_diagonal=False):
        graph = nx.Graph()
        map_matrix = self.grids
        
        for i in range(self.map_size):
            for j in range(self.map_size):
                if map_matrix[i][j]:
                    graph.add_node((i, j))
                    graph.add_edge((i, j), (i-1, j)) if self.if_connectable([i-1, j]) else None
                    graph.add_edge((i, j), (i, j-1)) if self.if_connectable([i, j-1]) else None
                    graph.add_edge((i, j), (i, j+1)) if self.if_connectable([i, j+1]) else None
                    graph.add_edge((i, j), (i+1, j)) if self.if_connectable([i+1, j]) else None
                    if if_diagonal:
                        graph.add_edge((i, j), (i-1, j-1)) if self.if_connectable([i-1, j-1]) else None
                        graph.add_edge((i, j), (i-1, j+1)) if self.if_connectable([i-1, j+1]) else None
                        graph.add_edge((i, j), (i+1, j-1)) if self.if_connectable([i+1, j-1]) else None
                        graph.add_edge((i, j), (i+1, j-1)) if self.if_connectable([i+1, j-1]) else None
    
        num_subgraph = len(list(nx.connected_components(graph)))
        if num_subgraph > 1:
            cprint("Not Connected !!! ", color='red', attrs=['bold'])        
            output = None
            max_subgraph_size = 0
            for idx, c in enumerate(nx.connected_components(graph)):
                print("subgraph ", idx, " size : ", graph.subgraph(c).number_of_nodes())
                if graph.subgraph(c).number_of_nodes() > max_subgraph_size:
                    max_subgraph_size = graph.subgraph(c).number_of_nodes()
                    output = graph.subgraph(c)
            cprint("Returing the Max Connected Component {} / {} !!! ".format(output.number_of_nodes(), graph.number_of_nodes()), color='red', attrs=['bold'])      
            self.load_from_networkx(output)
            self.draw_grid()
        else:
            output = graph
            cprint("Returing the Graph !!! ".format(output.number_of_nodes()), color='green', attrs=['bold'])      
        
        return output
    
    def if_connectable(self, pt):
        if pt[0] >= 0 and pt[0] <= self.map_size - 1 and pt[1] >=0 and pt[1] <= self.map_size - 1: 
            if self.grids[pt[0]][pt[1]]:
                return True
        return False

    def draw_grid(self):
        for i in range(self.map_size):
            for j in range(self.map_size):
                # position
                x = j * self.grid_size + self.map_margin
                y = i * self.grid_size + self.map_margin
                # size
                w = self.grid_size - 2 * self.map_margin
                h = self.grid_size - 2 * self.map_margin
                # state
                if self.grids[i][j] == 0:
                    color = self.obs_color
                else:
                    color = self.free_color
                pygame.draw.rect(self.window, color, (x, y, w, h))

    def render_graph(self, save_path, graph_name, padding=1, edge_length=30):
        graph_img = np.zeros(((self.map_size + padding) * edge_length, (self.map_size + padding) * edge_length, 3))
        graph = self.to_graph()
        for idx, nx_node in enumerate(graph.nodes):            
            pos =  np.array(nx_node)          
            cv2.circle(graph_img, 
               ((pos + padding)[::-1] * edge_length).astype(np.int32), 
               int(15), 
               (255,255,255), 
               -1)
            
        for idx, nx_edge in enumerate(graph.edges):
            start_pt = np.array(nx_edge[0])
            end_point = np.array(nx_edge[1])
            cv2.line(graph_img, 
             ((start_pt + padding)[::-1] * edge_length).astype(np.int32), 
             ((end_point + padding)[::-1] * edge_length).astype(np.int32), 
             (255,255,255), 
             8)  
        
        cv2.imwrite(save_path + graph_name + '.png', graph_img * 255)
        return graph_img
    
    def render_graph_curr(self, save_path, graph_name, padding=1, edge_length=30):
        img_size = (self.map_size + padding) * edge_length
        graph_img = np.zeros((img_size, img_size, 3))
        
        graph = self.to_graph()
        curr_node_idx = random.randint(0, graph.number_of_nodes() - 1)
        curr_node_idx_pos = np.array(list(graph.nodes)[curr_node_idx])
                
            
        for idx, nx_edge in enumerate(graph.edges):
            start_pt = (np.array(nx_edge[0]) - curr_node_idx_pos) / 2 + self.map_size / 2
            end_point = np.array(nx_edge[1] - curr_node_idx_pos) / 2 + self.map_size / 2
            cv2.line(graph_img, 
             ((start_pt + padding)[::-1] * edge_length).astype(np.int32), 
             ((end_point + padding)[::-1] * edge_length).astype(np.int32), 
             (255,255,255), 
             4)   
            
        for idx, nx_node in enumerate(graph.nodes): 
            pos =  (np.array(nx_node) - curr_node_idx_pos) / 2 + self.map_size / 2
            color = np.array([255,255,255])/255 if idx != curr_node_idx else np.array([0, 0, 255])/255
            cv2.circle(graph_img, 
               ((pos + padding)[::-1] * edge_length).astype(np.int32), 
               int(6), 
               color, 
               -1)
                
        cv2.imwrite(save_path + graph_name + '_curr.png', graph_img * 255)
        return graph_img

    def handle_mouse_click(self):
        # get mouse state
        x, y = pygame.mouse.get_pos()
        buttons = pygame.mouse.get_pressed()
        if buttons[0]: # left button clicked
            # pos
            i = y // self.grid_size
            j = x // self.grid_size
            # state switched
            self.grids[i][j] ^= 1

    def save(self, save_path, filename=None):
        if filename is None:
            filename = save_path + time.strftime("%Y%m%d%H%M%S")
        else:
            filename = save_path + filename
        pygame.image.save(self.window, filename + ".png")
        np.save(filename + ".npy", self.grids)
        cprint("Grid Map Saved : " + filename, color='green', attrs=['bold'])

    def run_plot(self, save_path=None):
        cprint("Tap Space to Save and Quit, Esc to Directly Quit", color='yellow', attrs=['bold'])
        running = True
        while running:
            for event in pygame.event.get():
                # Quit
                if event.type == pygame.QUIT:
                    running = False
                # Mouse
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_click()
                # Keyboard
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        graph = self.to_graph()
                        self.save(save_path)
                        running = False
            self.draw_grid()
            pygame.display.flip()
        pygame.quit()



def main():
    save_path = "../../graph_data/test/"
    # Mode 1: Start from Blank
    # a = GraphMapGenerator(map_size=20)
    # a.run_plot(save_path)

    # Mode 2: Read from Array
    # map_name = "map3"
    # array = np.load(save_path + map_name + ".npy")
    # a = GraphMapGenerator(array=array)
    # a.draw_grid()
    # a.save(save_path, map_name)
    # a.run_plot(save_path)
    
    # Mode 3: Read from Image
    map_name = "map_2"
    img = cv2.imread(save_path + map_name + ".png")
    a = GraphMapGenerator(img=img)
    a.run_plot(save_path)
    
    # To Graph
    # map_name = "map1"
    # array = np.load(save_path + map_name + ".npy")
    # a = GraphMapGenerator(array=array)
    # graph_img = a.render_graph(save_path, graph_name='test')
    # graph_img_curr = a.render_graph_curr(save_path, graph_name='test')
    
    
    


if __name__ == '__main__':
    main()