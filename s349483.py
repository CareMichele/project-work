from Problem import Problem
import numpy as np
import networkx as nx
import random

def get_problem_data(p):
    num_cities = len(p.graph.nodes)
    
    #gold_values[i] darà l'oro della città i 
    gold_values = np.array([p.graph.nodes[i]['gold'] for i in range(num_cities)])
    
    #inizializzo matrice vuota
    dist_matrix = np.zeros((num_cities, num_cities))
    
    #restituisce iteratore di (sorgente, {dest: distanza})
    path_iterator = nx.all_pairs_dijkstra_path_length(p.graph, weight='dist')
    
    #se le città sono tante, FW è lento, usiamo Dijkstra
    for source, dist_map in path_iterator:
        for dest, dist in dist_map.items():
            dist_matrix[source][dest] = dist
            
    return num_cities, gold_values, dist_matrix

def greedy_from_base(num_cities, dist_matrix):
    """Greedy partendo sempre dalla base"""
    unvisited = set(range(1, num_cities))
    tour = []
    current = 0
    
    while unvisited:
        next_city = min(unvisited, key=lambda c: dist_matrix[current][c])
        
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    
    return np.array(tour)

def greedy_nearest_neighbor(num_cities, dist_matr):
    """
    Greedy partendo da una città casuale.
    Utile per creare diversità nella popolazione mantenendo buoni tratti locali.
    """
    unvisited = set(range(1,num_cities))
    
    #nodo di partenza a caso
    start_node = random.choice(list(unvisited))
    
    tour = [start_node]
    unvisited.remove(start_node)
    current_node = start_node
    
    #nearest neighbor
    while unvisited:
        #trova il vicino più prossimo usando la matrice distanze
        next_node = min(unvisited, key = lambda c: dist_matr[current_node][c])
        
        tour.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node
        
    return np.array(tour)

def initialize_population(num_cities, pop_size, dist_matrix):
    population=[]
    
    #la base non viene inclusa
    base_tour = np.arange(1,num_cities)
    
    n_greedy = int(pop_size * 0.2)
    
    for i in range(pop_size):
        if i==0:
            individual = greedy_from_base(num_cities, dist_matrix)
        #hybrid initialization (20% hybrid, 80% random)
        elif i < n_greedy:
            individual = greedy_nearest_neighbor(num_cities, dist_matrix)
        else:
            individual = base_tour.copy()
            np.random.shuffle(individual)
            
        population.append(individual)
        
    return population

def calculate_fitness(ind, gold_values, dist_matrix, alpha, beta):
    current_node = 0
    current_gold = 0
    total_cost = 0
    
    for next_node in ind:
        dist_direct = dist_matrix[current_node][next_node] #km per andare diretti
        dist_to_base = dist_matrix[current_node][0] #km per tornare a casa
        dist_from_base = dist_matrix[0][next_node] #km da casa alla prossima
        
        gold_at_next= gold_values[next_node] #oro che trovo la
        
        #opzione 1: vado dritto
        cost_direct = dist_direct + (dist_direct * alpha * current_gold)**beta
        
        #opzione 2: passo dalla base
        cost_return = dist_to_base + (dist_to_base * alpha * current_gold)**beta
        cost_leave = dist_from_base
        cost_via_base = cost_return + cost_leave
        
        #confronto
        if cost_via_base < cost_direct:
            #conviene scaricare
            total_cost+=cost_via_base
            current_gold = gold_at_next
            #concettualmente abbiamo fatto current -> 0 -> next
        else:
            #conviene andare avanti
            total_cost+=cost_direct
            current_gold+=gold_at_next #aggiungo nuovo ora a quello vecchio
            
        current_node = next_node
    
    #fine giro (bisogna tornare alla base)
    dist_final = dist_matrix[current_node][0]
    total_cost+=dist_final+(dist_final*alpha*current_gold)**beta
    
    return total_cost

def tournament_selection(population, tau = 3):
    pool = random.sample(population, k=tau)
    return min(pool, key=lambda x: x[0])

def crossover(p1, p2):
    size = len(p1)
    
    l1 = np.random.randint(0,size)
    l2 = np.random.randint(0,size)
    if l1 > l2:
        l1,l2 = l2,l1
        
    child = np.full(size,-1)
    
    child[l1:l2+1] = p1[l1:l2+1]
    
    #cycle crossover (pag. 104 pdf 05)
    pos=0
    for i in p2:
        if i not in child:
            while l1 <= pos <= l2:
                pos+=1
                if pos >= size:
                    pos=0
            child[pos] = i
            pos+=1
            if pos >= size:
                pos = 0
            
    return child

def mutate(individual, mutation_rate):
    if np.random.random() < mutation_rate:
        size = len(individual)
        
        #scramble mutation (pag.99 pdf 05)
        l1 = np.random.randint(0,size)
        l2 = np.random.randint(0,size)
        #new_tour = p.tour[:]
        if l1>l2:
            l1,l2 = l2,l1
            
        segment = individual[l1:l2+1].copy()
        np.random.shuffle(segment)
        individual[l1:l2+1] = segment
        
    return individual

def reconstruct_path(ind, gold_values, dist_matrix,  alpha, beta, graph):
    """
    Costruisce il percorso FISICO completo.
    Se tra A e B non c'è una strada diretta, inserisce i nodi di transito
    trovati con lo shortest path di NetworkX.
    """
    full_path = []
    current_node = 0
    current_gold = 0.0
    
    # Helper locale per aggiungere segmenti di percorso fisico
    def add_segment(start, end, collect_gold_at_end=False):
        # Chiediamo a NetworkX la strada fisica (nodi intermedi)
        # Esempio: start=0, end=5 -> physical_nodes=[0, 2, 4, 5]
        physical_nodes = nx.shortest_path(graph, source=start, target=end, weight='dist')
        
        # Saltiamo il primo nodo perché è 'current_node' (già aggiunto o è lo start)
        for node in physical_nodes[1:]:
            g = 0 # Di base transitiamo (0 gold raccolto)
            
            # Se siamo arrivati alla destinazione e dovevamo raccogliere
            if node == end and collect_gold_at_end:
                g = gold_values[end]
            
            full_path.append((node, g))

    for next_node in ind:
        # --- LOGICA DI DECISIONE (identica alla fitness) ---
        dist_direct = dist_matrix[current_node][next_node]
        dist_to_base = dist_matrix[current_node][0]
        
        gold_at_next = gold_values[next_node]
        
        # Calcolo costi ipotetici
        cost_direct = dist_direct + (dist_direct * alpha * current_gold) ** beta
        
        cost_return = dist_to_base + (dist_to_base * alpha * current_gold) ** beta
        cost_leave = dist_matrix[0][next_node] 
        cost_via_base = cost_return + cost_leave
        
        if cost_via_base < cost_direct:
            # --- SCELTA: PASSO DALLA BASE ---
            
            # 1. Vado fisicamente alla base (se non ci sono già)
            if current_node != 0:
                add_segment(current_node, 0, collect_gold_at_end=False)
                # Arrivato a 0, ho scaricato implicitamente
            
            # 2. Riparto dalla base verso next_node
            add_segment(0, next_node, collect_gold_at_end=True)
            
            current_gold = gold_at_next # Ho solo l'oro nuovo
            
        else:
            # --- SCELTA: VADO DIRETTO ---
            add_segment(current_node, next_node, collect_gold_at_end=True)
            current_gold += gold_at_next # Accumulo
            
        current_node = next_node

    # Alla fine devo tornare alla base fisicamente
    if current_node != 0:
        add_segment(current_node, 0, collect_gold_at_end=False)
        
    return full_path



def solution(p:Problem):
    #estraggo i dati utili dal problema p 
    alpha = p.alpha
    beta = p.beta
    num_cities, gold_values, dist_matrix = get_problem_data(p)
    
    #EA 
    POP_SIZE = max(100, int(10*np.sqrt(num_cities)))
    MAX_GENERATIONS = max(200, int(20*np.sqrt(num_cities)))
    MUTATION_RATE = 0.2
    OFFSPRING_SIZE = int(POP_SIZE*0.5)
    
    raw_population = initialize_population(num_cities, POP_SIZE, dist_matrix)
    #for i in range(5):
     #   print(f"individuo {i}: {population[i]}")
     
    #Convertiamo la popolazione in una lista di tuple: (fitness, individual)
    # Calcoliamo la fitness una volta sola all'inizio
    population = []
    for tour in raw_population:
        fit = calculate_fitness(tour, gold_values, dist_matrix, alpha, beta)
        population.append((fit, tour))
    
    population.sort(key=lambda x: x[0])
    best_fitness = population[0][0]
    best_tour = population[0][1]
        
    step = 0
    no_improv = 0
    while step < MAX_GENERATIONS:
        offspring = []
        # Generiamo i figli
        for _ in range(OFFSPRING_SIZE):
            # NOTA: Nel tuo codice originale facevi O mutazione O crossover.
            # È meglio fare Crossover E POI Mutazione sul risultato.
            
            # A. Selezione Genitori (Tournament)
            p1_tuple = tournament_selection(population)
            p2_tuple = tournament_selection(population)
            p1, p2 = p1_tuple[1], p2_tuple[1] # Estraiamo solo i tour
            
            # B. Crossover (sempre, o con alta probabilità)
            if np.random.random() < 0.8:
                child_tour = crossover(p1, p2)
            else:
                child_tour = p1.copy()
            
            # C. Mutazione
            child_tour = mutate(child_tour, MUTATION_RATE)
            
            # Calcolo fitness del figlio
            child_fit = calculate_fitness(child_tour, gold_values, dist_matrix, alpha, beta)
            
            offspring.append((child_fit, child_tour))
            
        # STEADY STATE: Aggiungi figli e tieni solo i migliori
        population.extend(offspring)
        
        # Ordina per fitness (crescente, perché minore è meglio)
        population.sort(key=lambda x: x[0])
        
        # Taglia la popolazione per tornare alla dimensione originale (sopravvivono i migliori)
        population = population[:POP_SIZE]
        
        # Controlla miglioramenti (Logica Adattiva)
        current_best_fit = population[0][0]
        
        if current_best_fit < best_fitness:
            best_fitness = current_best_fit
            best_tour = population[0][1]
            no_improv = 0
            # Se troviamo un miglioramento, resettiamo un po' la mutation rate verso il basso
            # per sfruttare la zona buona (exploitation)
            MUTATION_RATE = max(0.1, MUTATION_RATE * 0.9) 
        else:
            no_improv += 1
            
        # Gestione Stagnazione (Logica Adattiva tua)
        if no_improv > 20:
            # Siamo bloccati? Aumentiamo il caos!
            MUTATION_RATE = min(0.6, MUTATION_RATE * 1.5) # Ho messo max 0.6 per non rompere tutto
            # Reset counter parziale per non farla salire all'infinito subito
            no_improv = 10 
            
        step += 1
        
        # Opzionale: Stampa progresso ogni tanto
        if step % 100 == 0:
            print(f"Step {step} | Best: {best_fitness:.2f} | MutRate: {MUTATION_RATE:.2f}")

    # --- 5. FORMATTAZIONE RISULTATO ---
    # Dobbiamo ricostruire il percorso (città, oro) esplicito
    # Usiamo la logica della fitness ma salvando il path invece di sommare solo il costo
    
    final_path = reconstruct_path(best_tour, gold_values, dist_matrix,  alpha, beta, p.graph)
    return final_path

def check_solution_score(p: Problem, path):
    total_cost = 0
    current_node = 0
    current_weight = 0 # Peso che ho sul camion
    
    for next_node, collected_gold in path:
        # Ora next_node è sicuramente un vicino fisico di current_node
        # p.cost calcola il costo del movimento
        edge_cost = p.cost([current_node, next_node], current_weight)
        total_cost += edge_cost
        
        # Aggiorno posizione
        current_node = next_node
        
        # Aggiorno peso
        if current_node == 0:
            current_weight = 0 # Scarico
        else:
            current_weight += collected_gold # Carico (se collected_gold è 0, transito e basta)
            
    return total_cost

if __name__ == '__main__':
    p = Problem(100, density=0.2, alpha=2, beta=1, seed=42)
    
    print("-" * 50)
    print("CALCOLO BASELINE (Prof)...")
    baseline_cost = p.baseline()
    print(f"Costo Baseline: {baseline_cost:,.2f}")
    
    print("-" * 50)
    print("CALCOLO GENETIC ALGORITHM (Tuo)...")
    solution_path = solution(p)
    
    # Calcolo il costo della tua soluzione
    my_cost = check_solution_score(p, solution_path)
    print(f"Costo Tuo GA:   {my_cost:,.2f}")
    
    print("-" * 50)
    # CONFRONTO
    gap = baseline_cost - my_cost
    improvement = (gap / baseline_cost) * 100
    
    if my_cost < baseline_cost:
        print(f"✅ VITTORIA! Hai risparmiato {gap:,.2f} ({improvement:.2f}%)")
    else:
        print(f"❌ SCONFITTA. La baseline è migliore di {-gap:,.2f}")
        
    print(f"Lunghezza percorso: {len(solution_path)} tappe")
    
    
