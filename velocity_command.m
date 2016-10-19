function output = velocity_command(u, P)

NN = 0;
target = zeros(P.num_targets,3);
for i=1:P.num_targets
    target(i,:) = [u(1+NN) u(2+NN) u(3+NN)];
    NN = NN+3;
end

camera_x = u(1+NN);     % camera x location
camera_y = u(2+NN);     % camera y location
camera_z = u(3+NN);     % camera z location
camera_phi = u(4+NN);   % camera phi
camera_theta = u(5+NN); % camera theta
camera_psi = u(6+NN);   % camera psi
NN = NN+6;

pixel = zeros(P.num_targets,3);
for i=1:P.num_targets
    pixel(i,:) = [u(1+NN) u(2+NN) u(3+NN)];
    NN = NN+3;
end

target_vector = zeros(P.num_targets,3);
target_depth = zeros(P.num_targets,1);
for i=1:P.num_targets
    target_vector(i,:) = [target(i,1)-camera_x target(i,2)-camera_y target(i,3)-camera_z];
    target_depth(i) = norm(target_vector(i,:));
end

% ==========multiple targets==========

% constructing the Image Jacobian matrix
L = zeros(P.num_targets*2,6);
NN = 0;
for i=1:P.num_targets
    L(1+NN,:) = [-1/target_depth(i)      0       pixel(i,1)/target_depth(i)      pixel(i,1)*pixel(i,2)    -(1+pixel(i,1)^2)    pixel(i,2)];
    L(2+NN,:) = [   0      -1/target_depth(i)    pixel(i,2)/target_depth(i)      1+pixel(i,2)^2      -pixel(i,1)*pixel(i,2)   -pixel(i,1)];
    NN = NN+2;
end

% singular_value = svd(L)'
% condition_number = cond(L)

% % image jacobian matrix
% L = [-1/target_depth1   0   pixel_x1/target_depth1   pixel_x1*pixel_y1   -(1+pixel_x1^2)   pixel_y1;...
%        0   -1/target_depth1 pixel_y1/target_depth1   1+pixel_y1^2    -pixel_x1*pixel_y1   -pixel_x1;...
%      -1/target_depth2   0   pixel_x2/target_depth2   pixel_x2*pixel_y2   -(1+pixel_x2^2)   pixel_y2;...
%        0   -1/target_depth2 pixel_y2/target_depth2   1+pixel_y2^2    -pixel_x2*pixel_y2   -pixel_x2;...
%      -1/target_depth3   0   pixel_x3/target_depth3   pixel_x3*pixel_y3   -(1+pixel_x3^2)   pixel_y3;...
%        0   -1/target_depth3 pixel_y3/target_depth3   1+pixel_y3^2    -pixel_x3*pixel_y3   -pixel_x3 ];

% task function Jacobian for mean and other terms in eq(11)
mean_x = mean(pixel(:,1));
mean_y = mean(pixel(:,2));
tf_m_error = [mean_x; mean_y];  % mean task function error. m_desired = [0; 0]

m_dot_m = -P.J_m_pinv*(P.gamma_m*tf_m_error - P.tf_m_dot_desired + P.c_epsillon^2/(P.k*P.kappa_m)*tf_m_error);

if P.num_targets==1
    V_c = pinv(L)*m_dot_m;
elseif P.num_targets > 1
    % task function Jacobian for variance and other terms in eq(14)
    variance_x = var(pixel(:,1));
    variance_y = var(pixel(:,2));
    temp_jv = zeros(2,2*P.num_targets);
    NN = 0;
    for i=1:P.num_targets
        temp_jv(1,1+NN) = pixel(i,1)-mean_x;
        temp_jv(2,2+NN) = pixel(i,2)-mean_y;
        NN = NN+2;
    end
    J_v = 2/P.k*temp_jv;
    J_v_pinv = J_v'*inv(J_v*J_v');  % right psedoinverse of J_v
    tf_v_error = [variance_x; variance_y] - P.tf_v_desired;
    m_dot_v = -J_v_pinv*(P.gamma_v*tf_v_error - P.tf_v_dot_desired + P.c_v^2*P.c_epsillon^2/P.kappa_v*tf_v_error);
    
    V_c = pinv(L)*(m_dot_m + m_dot_v);
%     V_c = pinv(L)*(m_dot_m);
end

V_c = sat(V_c);
output = V_c;

% output = [0; 0; 0; 0; 0; 0];
end

function out = sat(u)
    v_x = u(1);
    v_y = u(2);
    v_z = u(3);
    w_x = u(4);
    w_y = u(5);
    w_z = u(6);
    
    if v_x > 20
        v_x = 20;
    elseif v_x < -20
        v_x = -20;
    end
    
    if v_y > 20
        v_y = 20;
    elseif v_y < -20
        v_y = -20;
    end

    if v_z > 20
        v_z = 20;
    elseif v_z < -20
        v_z = -20;
    end

    if w_x > 1
        w_x = 1;
    elseif w_x < -1
        w_x = -1;
    end

    if w_y > 1
        w_y = 1;
    elseif w_y < -1
        w_y = -1;
    end

    if w_z > 1
        w_z = 1;
    elseif w_z < -1
        w_z = -1;
    end
    
    out = [v_x; v_y; v_z; w_x; w_y; w_z];
end