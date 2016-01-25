<%def name="make_maze(structure, height, size_scaling)">
    % for i in xrange(len(structure)):
        % for j in xrange(len(structure[0])):
            % if str(structure[i][j]) == '1':
                <geom
                  pos='${j*size_scaling} ${i*size_scaling} ${height/2*size_scaling}'
                  size='${0.5*size_scaling} ${0.5*size_scaling} ${height/2*size_scaling}'
                  type='box'
                  material=""
                  rgba='0.4 0.4 0.4 1'
                  />
            % endif
        % endfor
    % endfor
</%def>

<%def name="find_robot(structure, size_scaling)">
    <%
        robot_pos = [0, 0]
        found = False
        for i in xrange(len(structure)):
            for j in xrange(len(structure[0])):
                if structure[i][j] == 'r':
                    robot_pos = [j*size_scaling, i*size_scaling]
                    found = True
                    break
            if found:
                break
    %>
    ${' '.join(map(str, robot_pos)) + " 0"}
</%def>
