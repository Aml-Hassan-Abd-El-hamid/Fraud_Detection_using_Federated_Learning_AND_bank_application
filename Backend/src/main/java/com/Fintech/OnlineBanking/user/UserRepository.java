package com.Fintech.OnlineBanking.user;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepository  extends JpaRepository<User, Long>{
Optional<User>findByUsername(String username);
User findByUserInfoNationalId(Long nationalId);

}
