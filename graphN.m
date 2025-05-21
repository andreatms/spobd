function A = graphN(num_clusters, nodes_per_cluster, p_intra, p_inter, weights_intra, weights_inter)
% GRAPHN Genera un grafo pesato con un numero parametrico di cluster
%
% Parametri:
%   num_clusters - Numero di cluster da generare
%   nodes_per_cluster - Vettore con il numero di nodi per ogni cluster
%                       (lunghezza deve essere uguale a num_clusters)
%   p_intra - Probabilità di connessione all'interno di ciascun cluster
%             (può essere un valore singolo o un vettore di lunghezza num_clusters)
%   p_inter - Matrice num_clusters x num_clusters che specifica
%             la probabilità di connessione tra coppie di cluster
%   weights_intra - Pesi possibili per gli archi intra-cluster
%                   (può essere un vettore singolo o una cella di lunghezza num_clusters)
%   weights_inter - Pesi possibili per gli archi inter-cluster
%                   (può essere un vettore singolo o una matrice di celle num_clusters x num_clusters)
%
% Output:
%   A - Matrice di adiacenza del grafo generato

% Valori di default per i pesi se non specificati
if nargin < 5 || isempty(weights_intra)
    weights_intra = {[1, 5, 10, 25, 50]};
end

if nargin < 6 || isempty(weights_inter)
    weights_inter = {[1, 5, 10, 25, 50]};
end

% Validazione input
if length(nodes_per_cluster) ~= num_clusters
    error('La lunghezza del vettore nodes_per_cluster deve essere uguale a num_clusters');
end

if length(p_intra) == 1
    p_intra = p_intra * ones(1, num_clusters);
elseif length(p_intra) ~= num_clusters
    error('p_intra deve essere un singolo valore o un vettore di lunghezza num_clusters');
end

if ~all(size(p_inter) == [num_clusters, num_clusters])
    error('p_inter deve essere una matrice di dimensioni num_clusters x num_clusters');
end

% Gestione dei pesi intra-cluster
if ~iscell(weights_intra)
    % Se è un array normale, lo converte in una cella singola
    weights_intra = {weights_intra};
end

if length(weights_intra) == 1
    % Se c'è un solo set di pesi, lo replica per tutti i cluster
    weights_intra = repmat(weights_intra, 1, num_clusters);
elseif length(weights_intra) ~= num_clusters
    error('weights_intra deve essere un singolo set di pesi o una cella di lunghezza num_clusters');
end

% Gestione dei pesi inter-cluster
if ~iscell(weights_inter)
    % Se è un array normale, lo converte in una cella singola
    weights_inter = {weights_inter};
end

if length(weights_inter) == 1
    % Se c'è un solo set di pesi, crea una matrice di celle
    cell_matrix = cell(num_clusters);
    for i = 1:num_clusters
        for j = 1:num_clusters
            cell_matrix{i,j} = weights_inter{1};
        end
    end
    weights_inter = cell_matrix;
elseif ~iscell(weights_inter) || ~all(size(weights_inter) == [num_clusters, num_clusters])
    error('weights_inter deve essere un singolo set di pesi o una matrice di celle num_clusters x num_clusters');
end

% Calcolo numero totale di nodi
total_nodes = sum(nodes_per_cluster);

% Inizializzazione della matrice di adiacenza
A = zeros(total_nodes);

% Calcolo degli indici di partenza per ogni cluster
start_indices = zeros(1, num_clusters);
for c = 2:num_clusters
    start_indices(c) = start_indices(c-1) + nodes_per_cluster(c-1);
end

% Generazione degli archi all'interno di ciascun cluster
for c = 1:num_clusters
    start_idx = start_indices(c) + 1;
    end_idx = start_indices(c) + nodes_per_cluster(c);
    
    for i = start_idx:end_idx
        for j = i+1:end_idx
            if rand < p_intra(c)
                % Usa i pesi specifici per questo cluster
                weight = weights_intra{c}(randi(length(weights_intra{c})));
                A(i,j) = weight;
                A(j,i) = weight;
            end
        end
    end
end

% Aggiunta degli archi ponte tra cluster
for c1 = 1:num_clusters
    for c2 = c1+1:num_clusters
        if p_inter(c1, c2) > 0
            c1_start = start_indices(c1) + 1;
            c1_end = start_indices(c1) + nodes_per_cluster(c1);
            c2_start = start_indices(c2) + 1;
            c2_end = start_indices(c2) + nodes_per_cluster(c2);
            
            for i = c1_start:c1_end
                for j = c2_start:c2_end
                    if rand < p_inter(c1, c2)
                        % Usa i pesi specifici per questa coppia di cluster
                        weight = weights_inter{c1,c2}(randi(length(weights_inter{c1,c2})));
                        A(i,j) = weight;
                        A(j,i) = weight;
                    end
                end
            end
        end
    end
end
end